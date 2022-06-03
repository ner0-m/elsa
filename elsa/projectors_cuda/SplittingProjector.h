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
            // const DetectorDescriptor& detectorDescriptor =
            // static_cast<const DetectorDescriptor&>(*_rangeDescriptor);
            const RealVector_t detSize =
                _rangeDescriptor->getNumberOfCoefficientsPerDimension().template cast<real_t>();

            RealVector_t maximal = RealVector_t::Zero(3);
            RealVector_t sliceStart = RealVector_t::Zero(2);
            RealVector_t sliceEnd = detSize.topRows(2);
            index_t numPoses = 0;
            for (const auto& I : poses)
                numPoses += I.second - I.first;

            if (numPoses < 50)
                bucket = NUM_BUCKETS - 1;

            if (bucket != NUM_BUCKETS - 1) {
                index_t imgAxis = bucket % 6 / 3;
                cudaDeviceProp props;
                cudaGetDeviceProperties(&props, 0);
                const int maxBlockDim = 8;

                real_t SLICE_THICKNESS = std::max(
                    maxBlockDim,
                    static_cast<int>(std::ceil(static_cast<real_t>(props.maxThreadsPerMultiProcessor
                                                                   * props.multiProcessorCount)
                                               / detSize[1 - imgAxis] / maxBlockDim)
                                     * maxBlockDim));

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
                        intervals.emplace(iStart - otherStart,
                                          std::tuple(otherStart, iStart, otherBucket));
                        it2 = intervals.erase(it2);
                    } else if (otherStart >= iStart && otherStart < iEnd) {
                        // partial overlap with beginning of smaller interval
                        intervals.emplace(otherEnd - iEnd, std::tuple(iEnd, otherEnd, otherBucket));
                        it2 = intervals.erase(it2);
                    } else {
                        it2++;
                    }
                }
            }

            // determine chunks
            std::array<std::vector<Interval>, NUM_BUCKETS - 1> intervalVecs;
            for (const auto& [size, iData] : intervals) {
                const auto& [iStart, iEnd, bucket] = iData;
                intervalVecs[bucket].emplace_back(iStart, iEnd);
            }
            for (index_t i = 0; i < intervalVecs.size(); i++) {
                if (!intervalVecs[i].empty()) {
                    std::sort(intervalVecs[i].begin(), intervalVecs[i].end());

                    numPoses = 0;
                    for (const auto& I : intervalVecs[i])
                        numPoses += I.second - I.first;
                    if (numPoses >= 50) {
                        determineChunksForPoses(intervalVecs[i], i);
                    } else {
                        highlyRotated.insert(highlyRotated.end(), intervalVecs[i].begin(),
                                             intervalVecs[i].end());
                    }
                }
            }

            if (!highlyRotated.empty()) {
                std::sort(highlyRotated.begin(), highlyRotated.end());
                determineChunksForPoses(highlyRotated, NUM_BUCKETS - 1);
            }

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
              _partitioningForward{}
        {
            int activeDevice;
            gpuErrchk(cudaGetDevice(&activeDevice));
            for (const int& targetDevice : targetDevices) {
                gpuErrchk(cudaSetDevice(targetDevice));
                _projectors.push_back(
                    std::make_unique<ProjectionMethod>(volumeDescriptor, detectorDescriptor, true));
            }
            gpuErrchk(cudaSetDevice(activeDevice));

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

            const auto AxFlags = cudaHostRegisterPortable;
            const auto xFlags = AxFlags | cudaHostRegisterReadOnly;
            gpuErrchk(cudaHostRegister((void*) &x[0], x.getSize() * sizeof(data_t), xFlags));
            gpuErrchk(cudaHostRegister((void*) &Ax[0], Ax.getSize() * sizeof(data_t), AxFlags));

            // TODO: look at different handling orders e.g. slice from middle of container
            // followed by slice from side
            std::array<std::vector<int>, NUM_BUCKETS> slices;
            for (int i = 0; i < _maxChunkForward.size(); i++)
                slices[i] = makeIndexSequence(_partitioningForward[i].size());
            std::mutex sliceLock;

            std::vector<std::thread> schedulers(0);

            index_t maxChunkDomain = 0;
            index_t maxChunkRange = 0;
            for (int j = 1; j < _maxChunkForward.size(); j++) {
                if (_maxChunkForward[j].first.prod()
                    > _maxChunkForward[maxChunkDomain].first.prod())
                    maxChunkDomain = j;
                if (_maxChunkForward[j].second.prod()
                    > _maxChunkForward[maxChunkRange].second.prod())
                    maxChunkRange = j;
            }

            auto scheduler = [this](const DataContainer<data_t>& x, DataContainer<data_t>& Ax,
                                    std::array<std::vector<int>, NUM_BUCKETS>& slices,
                                    std::mutex& sliceLock, std::size_t i, index_t maxChunkDomain,
                                    index_t maxChunkRange, std::size_t freeMemory) {
                gpuErrchk(cudaSetDevice(_projectors[i]->getDevice()));

                index_t arrMaxElements = freeMemory / (2 * sizeof(data_t));
                // create a cudaStream for this thread
                const auto stream = std::make_unique<CudaStreamWrapper>();

                IndexVector_t domainMaxDims = IndexVector_t::Ones(3);
                IndexVector_t rangeMaxDims = IndexVector_t::Ones(3);
                for (int j = 0; j < _maxChunkForward.size(); j++) {
                    if (slices[j].empty())
                        continue;
                    const auto& [d, r] = _maxChunkForward[j];
                    domainMaxDims = (d.array() > domainMaxDims.array()).select(d, domainMaxDims);
                    if (r[0] > rangeMaxDims[0])
                        rangeMaxDims[0] = r[0];
                    if (r[1] * r[2] > rangeMaxDims[1])
                        rangeMaxDims[1] = r[1] * r[2];
                }
                real_t rowAlignment = static_cast<real_t>(512 / sizeof(data_t));
                domainMaxDims[0] = static_cast<index_t>(
                    std::ceil(static_cast<real_t>(domainMaxDims[0]) / rowAlignment) * rowAlignment);
                rangeMaxDims[0] = static_cast<index_t>(
                    std::ceil(static_cast<real_t>(rangeMaxDims[0]) / rowAlignment) * rowAlignment);
                const auto singleAllocSize = domainMaxDims.prod() + rangeMaxDims.prod();

                BoundingBox currentVolumeBox(IndexVector_t::Ones(3));
                IndexVector_t currentVolumeDims = IndexVector_t::Zero(3);
                IndexVector_t currentImageDims = IndexVector_t::Zero(3);
                std::unique_ptr<CUDAVariablesForward<data_t>> cudaVars;
                if (singleAllocSize <= freeMemory / sizeof(data_t)) {
                    cudaVars =
                        _projectors[i]->setupCUDAVariablesForward(domainMaxDims, rangeMaxDims);
                    cudaVars->stream = stream.get();

                    currentVolumeDims = domainMaxDims;
                    currentImageDims = rangeMaxDims;
                }

                for (int j = _maxChunkForward.size() - 1; j >= 0; j--) {
                    if (slices[j].empty())
                        continue;

                    const auto& tasks = _partitioningForward[j];

                    // determine dimensions of padded arrays
                    IndexVector_t volumeDims = _maxChunkForward[j].first;
                    volumeDims[0] = static_cast<index_t>(
                        std::ceil(static_cast<real_t>(volumeDims[0]) / rowAlignment)
                        * rowAlignment);
                    IndexVector_t imageDims = _maxChunkForward[j].second;
                    imageDims[0] = static_cast<index_t>(
                        std::ceil(static_cast<real_t>(imageDims[0]) / rowAlignment) * rowAlignment);
                    imageDims[1] *= imageDims[2];
                    imageDims[2] = 1;
                    // check if reallocation necessary
                    if ((currentVolumeDims.array() >= volumeDims.array()).all()
                        && (currentImageDims.array() >= imageDims.array()).all()) {
                        // no reallocation
                    } else {
                        // reallocate
                        const auto paddedVolSize = volumeDims.prod();
                        const auto paddedSinoSize = imageDims.prod();
                        index_t splitIdx =
                            _maxChunkForward[j].first[1] > _maxChunkForward[j].first[2] ? 1 : 2;
                        if (paddedVolSize > arrMaxElements) {
                            const auto numSubtasks = paddedVolSize / arrMaxElements + 1;
                            volumeDims[splitIdx] = static_cast<index_t>(
                                std::ceil(static_cast<real_t>(volumeDims[splitIdx])
                                          / static_cast<real_t>(numSubtasks)));
                        }
                        if (paddedSinoSize > arrMaxElements) {
                            const auto numSubtasks = paddedSinoSize / arrMaxElements + 1;
                            imageDims[1] =
                                static_cast<index_t>(std::ceil(static_cast<real_t>(imageDims[1])
                                                               / static_cast<real_t>(numSubtasks)));
                        }

                        cudaVars = _projectors[i]->setupCUDAVariablesForward(volumeDims, imageDims);
                        cudaVars->stream = stream.get();

                        currentVolumeDims = volumeDims;
                        currentImageDims = imageDims;
                    }

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

                        for (const auto& subtask : _projectors[i]->getSubtasks(
                                 task, currentVolumeDims, currentImageDims)) {
                            _projectors[i]->applyConstrained(x, Ax, subtask, *cudaVars);
                        }
                    }
                }
            };

            int activeDevice;
            gpuErrchk(cudaGetDevice(&activeDevice));
            for (std::size_t i = 0; i < _projectors.size(); i++) {
                std::size_t free, total;
                auto device = _projectors[i]->getDevice();
                gpuErrchk(cudaSetDevice(device));
                gpuErrchk(cudaMemGetInfo(&free, &total));

                schedulers.emplace_back(scheduler, std::cref(x), std::ref(Ax), std::ref(slices),
                                        std::ref(sliceLock), i, maxChunkDomain, maxChunkRange,
                                        free / 2);
                schedulers.emplace_back(scheduler, std::cref(x), std::ref(Ax), std::ref(slices),
                                        std::ref(sliceLock), i, maxChunkDomain, maxChunkRange,
                                        free / 2);
            }

            for (auto& scheduler : schedulers)
                scheduler.join();
            gpuErrchk(cudaSetDevice(activeDevice));
            gpuErrchk(cudaHostUnregister((void*) &x[0]));
            gpuErrchk(cudaHostUnregister((void*) &Ax[0]));
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