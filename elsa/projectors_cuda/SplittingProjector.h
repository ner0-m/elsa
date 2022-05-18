#pragma once

#include "CUDAProjector.h"
#include "Timer.h"

#include <cuda_runtime.h>

#include <iostream>
#include <utility>
#include <thread>
#include <mutex>
#include <map>
#include <math.h>

namespace elsa
{
    template <template <typename> typename T, typename data_t>
    constexpr data_t return_data_t(const T<data_t>&);

    template <typename T>
    using get_data_t = decltype(return_data_t(std::declval<T>()));

    template <typename ProjectionMethod>
    class SplittingProjector : public LinearOperator<get_data_t<ProjectionMethod>>
    {
    private:
        using data_t = get_data_t<ProjectionMethod>;
        static_assert(std::is_base_of_v<CUDAProjector<data_t>, ProjectionMethod>,
                      "ProjectionMethod must be a CUDAProjector");

        static constexpr index_t NUM_BUCKETS = 13;

    public:
        void determineChunksForPoses(const std::vector<Interval>& poses, index_t bucket)
        {
            constexpr real_t SLICE_THICKNESS = 32;

            // const DetectorDescriptor& detectorDescriptor =
            // static_cast<const DetectorDescriptor&>(*_rangeDescriptor);
            const RealVector_t detSize =
                _rangeDescriptor->getNumberOfCoefficientsPerDimension().template cast<real_t>();

            RealVector_t maximal = RealVector_t::Zero(3);
            RealVector_t sliceStart = RealVector_t::Zero(2);
            RealVector_t sliceEnd = detSize.topRows(2);
            if (bucket != NUM_BUCKETS - 1) {
                index_t imgAxis = bucket % 6 / 3;
                index_t numPoses = 0;
                for (const auto& I : poses)
                    numPoses += I.second - I.first;

                for (index_t slice = 0; slice * SLICE_THICKNESS < detSize[imgAxis]; slice++) {
                    sliceStart[imgAxis] = slice * SLICE_THICKNESS;
                    sliceEnd[imgAxis] = sliceStart[imgAxis] + SLICE_THICKNESS;
                    if (sliceEnd[imgAxis] > detSize[imgAxis])
                        sliceEnd[imgAxis] = detSize[imgAxis];

                    BoundingBox imagePatch(sliceStart, sliceEnd);

                    const auto volumeBox =
                        _projectors[0]->constrainProjectionSpace2(imagePatch, poses);
                    _partitioningForward[bucket].push_back(
                        {volumeBox, imagePatch, poses, numPoses});
                }
            } else {
                index_t numPoses = 0;
                for (const auto& I : poses)
                    numPoses += I.second - I.first;

                BoundingBox imagePatch(sliceStart, sliceEnd);
                const auto volumeBox = _projectors[0]->constrainProjectionSpace2(imagePatch, poses);
                _partitioningForward[bucket].push_back({volumeBox, imagePatch, poses, numPoses});
            }
        }

        void splitProblem()
        {
            Timer timeguard("SplittingProjector", "splitProblem");

            const auto& detectorDescriptor =
                static_cast<const DetectorDescriptor&>(*_rangeDescriptor);

            RealVector_t detectorCenter = detectorDescriptor.getNumberOfCoefficientsPerDimension()
                                              .head(2)
                                              .template cast<real_t>()
                                          / static_cast<real_t>(2);

            auto numPoses = detectorDescriptor.getNumberOfGeometryPoses();

            Eigen::Matrix<bool, -1, NUM_BUCKETS> axisAssignment =
                Eigen::Matrix<bool, -1, -1>::Constant(numPoses, NUM_BUCKETS, false);
            axisAssignment.col(NUM_BUCKETS - 1) =
                Eigen::Matrix<bool, -1, 1>::Constant(numPoses, true);
            const auto detSize = detectorDescriptor.getNumberOfCoefficientsPerDimension();
            IndexVector_t midRowLeft(2), midRowRight(2), midColTop(2), midColBottom(2);
            midRowLeft << detSize[0] / 2, 0;
            midRowRight << detSize[0] / 2 + 1, detSize[1];
            midColTop << 0, detSize[1] / 2;
            midColBottom << detSize[0], detSize[1] / 2 + 1;
            BoundingBox midRow(midRowLeft, midRowRight);
            BoundingBox midCol(midColTop, midColBottom);

            const RealVector_t volSize =
                _domainDescriptor->getNumberOfCoefficientsPerDimension().template cast<real_t>();

            const RealMatrix_t volFrame = RealMatrix_t::Identity(3, 3);

            // vector of intervals of highly rotated poses
            std::vector<std::pair<index_t, index_t>> highlyRotated;
            bool lastPoseHighlyRotated = false;
            for (index_t pose = 0; pose < numPoses; pose++) {
                const RealMatrix_t projVolFrame =
                    detectorDescriptor.getGeometryAt(pose)->getRotationMatrix().topRows(2);

                auto bb = _projectors[0]->constrainProjectionSpace2(
                    midRow, std::vector{Interval(pose, pose + 1)});
                RealVector_t bbDims = bb._max - bb._min;
                axisAssignment.block(pose, 0, 1, 3) =
                    ((bbDims.array() / volSize.array()) < static_cast<real_t>(1.0 / 3)).transpose()
                    && projVolFrame.row(0).array() > 0;
                axisAssignment.block(pose, 6, 1, 3) =
                    ((bbDims.array() / volSize.array()) < static_cast<real_t>(1.0 / 3)).transpose()
                    && projVolFrame.row(0).array() < 0;
                bb = _projectors[0]->constrainProjectionSpace2(
                    midCol, std::vector{Interval(pose, pose + 1)});
                bbDims = bb._max - bb._min;
                axisAssignment.block(pose, 3, 1, 3) =
                    ((bbDims.array() / volSize.array()) < static_cast<real_t>(1.0 / 3)).transpose()
                    && projVolFrame.row(1).array() > 0;
                axisAssignment.block(pose, 9, 1, 3) =
                    ((bbDims.array() / volSize.array()) < static_cast<real_t>(1.0 / 3)).transpose()
                    && projVolFrame.row(1).array() < 0;

                bool thisPoseHighlyRotated = !axisAssignment.block(pose, 0, 1, 12).array().any();
                if (thisPoseHighlyRotated && !lastPoseHighlyRotated) {
                    highlyRotated.emplace_back(pose, pose + 1);
                    lastPoseHighlyRotated = true;
                } else if (!thisPoseHighlyRotated && lastPoseHighlyRotated) {
                    highlyRotated.back().second = pose;
                    lastPoseHighlyRotated = false;
                }
            }
            if (lastPoseHighlyRotated) {
                highlyRotated.back().second = numPoses;
            }

            // multimap of intervals sorted by size, highly rotated poses are excluded
            // size -> (I_begin, I_end, bucket)
            std::multimap<index_t, std::tuple<index_t, index_t, index_t>, std::greater<index_t>>
                intervals;
            IndexVector_t intervalBegin = IndexVector_t::Constant(NUM_BUCKETS - 1, numPoses);
            for (index_t pose = 0; pose < numPoses; pose++) {
                for (index_t i = 0; i < NUM_BUCKETS - 1; i++) {
                    if (intervalBegin[i] == numPoses && axisAssignment(pose, i)) {
                        // beginning of interval
                        intervalBegin[i] = pose;
                    } else if (intervalBegin[i] < pose && !axisAssignment(pose, i)) {
                        // end of interval
                        intervals.emplace(pose - intervalBegin[i],
                                          std::tuple(intervalBegin[i], pose, i));
                        intervalBegin[i] = numPoses;
                    }
                }
            }
            for (index_t i = 0; i < NUM_BUCKETS - 1; i++) {
                if (intervalBegin[i] < numPoses && axisAssignment(numPoses - 1, i)) {
                    intervals.emplace(numPoses - intervalBegin[i],
                                      std::tuple(intervalBegin[i], numPoses, i));
                }
            }

            // TODO: this is a greedy first best algorithm, might not be optimal
            for (auto it = intervals.begin(); it != intervals.end(); it++) {
                const auto& [size, iData] = *it;
                const auto& [iStart, iEnd, bucket] = iData;

                auto it2 = it;
                for (it2++; it2 != intervals.end();) {
                    const auto& [sizeOther, otherData] = *it2;
                    const auto& [otherStart, otherEnd, otherBucket] = otherData;

                    if (otherStart >= iStart && otherEnd <= iEnd) {
                        // smaller interval fully contained in larger one
                        it2 = intervals.erase(it2);
                    } else if (otherEnd > iStart && otherEnd <= iEnd) {
                        // partial overlap with end of smaller interval
                        intervals.emplace(otherEnd - iStart,
                                          std::tuple(iStart, otherEnd, otherBucket));
                        it2 = intervals.erase(it2);
                    } else if (otherStart >= iStart && otherStart < iEnd) {
                        // partial overlap with beginning of smaller interval
                        intervals.emplace(iEnd - otherStart,
                                          std::tuple(otherStart, iEnd, otherBucket));
                        it2 = intervals.erase(it2);
                    } else {
                        it2++;
                    }
                }
            }
            std::vector<std::tuple<index_t, index_t, index_t>> smallIntervals;

            // determine chunks
            std::array<std::vector<Interval>, NUM_BUCKETS - 1> intervalVecs;
            for (const auto& [size, iData] : intervals) {
                const auto& [iStart, iEnd, bucket] = iData;
                intervalVecs[bucket].emplace_back(iStart, iEnd);
            }
            for (index_t i = 0; i < intervalVecs.size(); i++) {
                if (!intervalVecs[i].empty())
                    determineChunksForPoses(intervalVecs[i], i);
            }

            determineChunksForPoses(highlyRotated, NUM_BUCKETS - 1);

            for (index_t bucket = 0; bucket < NUM_BUCKETS; bucket++) {
                auto& maxChunk = _maxChunkForward[bucket];
                const auto& tasks = _partitioningForward[bucket];
                if (!tasks.empty()) {
                    maxChunk.first = (tasks[0]._volumeBox._max - tasks[0]._volumeBox._min)
                                         .template cast<index_t>();
                    maxChunk.second = IndexVector_t::Constant(3, tasks[0]._numPoses);
                    maxChunk.second.topRows(2) =
                        (tasks[0]._imagePatch._max - tasks[0]._imagePatch._min)
                            .template cast<index_t>();

                    for (index_t i = 1; i < tasks.size(); i++) {
                        IndexVector_t volSize =
                            (tasks[i]._volumeBox._max - tasks[i]._volumeBox._min)
                                .template cast<index_t>();
                        IndexVector_t sinoSize = IndexVector_t::Constant(3, tasks[i]._numPoses);
                        sinoSize.topRows(2) =
                            (tasks[i]._imagePatch._max - tasks[i]._imagePatch._min)
                                .template cast<index_t>();

                        maxChunk.first = (maxChunk.first.array() >= volSize.array())
                                             .select(maxChunk.first, volSize);
                        maxChunk.second = (maxChunk.second.array() >= sinoSize.array())
                                              .select(maxChunk.second, sinoSize);
                    }
                }
            }
        }

        SplittingProjector(
            const VolumeDescriptor& volumeDescriptor, const DetectorDescriptor& detectorDescriptor,
            const std::vector<int> targetDevices = makeIndexSequence(numAvailableDevices()))
            : LinearOperator<data_t>{volumeDescriptor, detectorDescriptor},
              _projectors(0),
              _partitioningForward{{}, {}, {}, {}}
        {
            for (const int& targetDevice : targetDevices) {
                _projectors.push_back(std::make_unique<ProjectionMethod>(
                    volumeDescriptor, detectorDescriptor, true, targetDevice));
            }

            splitProblem();
        }

        ~SplittingProjector() override = default;

        SplittingProjector<ProjectionMethod>&
            operator=(SplittingProjector<ProjectionMethod>&) = delete;

    protected:
        SplittingProjector(const SplittingProjector<ProjectionMethod>& other)
            : LinearOperator<data_t>(other)
        {
        }

        /// apply forward projection
        void applyImpl(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const override
        {
            Timer timeguard("SplittingProjector", "apply");

            // TODO: look at different handling orders e.g. slice from middle of container
            // followed by slice from side
            std::array<std::vector<int>, NUM_BUCKETS> slices;
            for (int i = 0; i < _maxChunkForward.size(); i++)
                slices[i] = makeIndexSequence(_partitioningForward[i].size());
            std::mutex sliceLock;

            std::vector<std::thread> schedulers(0);

            auto scheduler = [this](const DataContainer<data_t>& x, DataContainer<data_t>& Ax,
                                    std::array<std::vector<int>, NUM_BUCKETS>& slices,
                                    std::mutex& sliceLock, std::size_t i) {
                for (int j = 0; j < _maxChunkForward.size(); j++) {
                    if (slices[j].empty())
                        continue;

                    const auto& tasks = _partitioningForward[j];

                    auto cudaVars = _projectors[i]->setupCUDAVariablesForwardConstrained(
                        _maxChunkForward[j].first, _maxChunkForward[j].second);

                    int sliceNum;
                    while (true) {
                        // select next slice
                        {
                            std::scoped_lock lock(sliceLock);

                            if (slices[j].empty())
                                break;

                            sliceNum = slices[j][slices[j].size() - 1];
                            slices[j].pop_back();
                        }

                        const auto& task = tasks[sliceNum];

                        _projectors[i]->applyConstrained(x, Ax, task, *cudaVars);
                    }
                }
            };

            for (std::size_t i = 0; i < _projectors.size(); i++) {
                schedulers.emplace_back(scheduler, std::cref(x), std::ref(Ax), std::ref(slices),
                                        std::ref(sliceLock), i);
                schedulers.emplace_back(scheduler, std::cref(x), std::ref(Ax), std::ref(slices),
                                        std::ref(sliceLock), i);
            }

            for (auto& scheduler : schedulers)
                scheduler.join();
        }

        /// apply adjoint of forward projection (i.e. backward projection)
        void applyAdjointImpl(const DataContainer<data_t>& y,
                              DataContainer<data_t>& Aty) const override
        {
        }

        /// implement the polymorphic clone operation
        SplittingProjector<ProjectionMethod>* cloneImpl() const override {}

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override {}

    private:
        std::vector<std::unique_ptr<ProjectionMethod>> _projectors;

        std::vector<ForwardProjectionTask> _partitioningForward[NUM_BUCKETS];
        std::array<std::pair<IndexVector_t, IndexVector_t>, NUM_BUCKETS> _maxChunkForward;

        static int numAvailableDevices()
        {
            int deviceCount;
            cudaGetDeviceCount(&deviceCount);
            return deviceCount;
        }

        static std::vector<int> makeIndexSequence(std::size_t n)
        {
            std::vector<int> indexSequence(n);

            for (std::size_t i = 0; i < n; i++)
                indexSequence[i] = i;

            return indexSequence;
        }

        /// lift from base class
        using LinearOperator<data_t>::_domainDescriptor;
        using LinearOperator<data_t>::_rangeDescriptor;
    };
} // namespace elsa