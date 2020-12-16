#pragma once

#include "CUDAProjector.h"
#include "Timer.h"

#include <cuda_runtime.h>

#include <iostream>
#include <utility>
#include <thread>
#include <mutex>

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

    public:
        SplittingProjector(
            const VolumeDescriptor& volumeDescriptor, const DetectorDescriptor& detectorDescriptor,
            const std::vector<int> targetDevices = makeIndexSequence(numAvailableDevices()))
            : LinearOperator<data_t>{volumeDescriptor, detectorDescriptor},
              _projectors(0),
              _partitioningForward{}
        {
            index_t sliceThickness = 16;

            for (const int& targetDevice : targetDevices) {
                _projectors.push_back(std::make_unique<ProjectionMethod>(
                    volumeDescriptor, detectorDescriptor, true, targetDevice));
            }

            const auto sinoSize = detectorDescriptor.getNumberOfCoefficientsPerDimension();

            IndexVector_t countMainDir =
                detectorDescriptor.getCountOfPrincipalRaysPerMainDirection();

            // axis of rotation
            index_t aor;
            countMainDir.minCoeff(&aor);
            RealVector_t aorVecVol = RealVector_t::Zero(3);
            aorVecVol[aor] = 1;
            Logger::get("SplittingProjector")->debug("AOR is {}", aor);

            // determine image split axis
            index_t splitAxis;
            const RealVector_t aorVecDet =
                detectorDescriptor.getGeometryAt(0)->getRotationMatrix() * aorVecVol;

            if (std::abs(aorVecDet[0]) >= std::abs(aorVecDet[1])) {
                splitAxis = 0;
            } else {
                splitAxis = 1;
            }
            Logger::get("SplittingProjector")->debug("Splitting along image axis {}", splitAxis);

            RealVector_t maximal = RealVector_t::Zero(3);
            for (index_t i = 0; i < sinoSize[splitAxis]; i += sliceThickness) {
                IndexVector_t startCoordinate = IndexVector_t::Zero(3);
                startCoordinate[splitAxis] = i;
                IndexVector_t endCoordinate = sinoSize;
                if (i + sliceThickness <= sinoSize[splitAxis])
                    endCoordinate[splitAxis] = i + sliceThickness;

                const BoundingBox sinoBox(startCoordinate, endCoordinate);
                const auto volumeBox = _projectors[0]->constrainProjectionSpace(sinoBox);

                _partitioningForward.emplace_back(volumeBox, sinoBox);

                RealVector_t size = volumeBox._max - volumeBox._min;
                maximal = (size.array() > maximal.array()).select(size, maximal);
            }

            for (auto& [volumeBox, sinoBox] : _partitioningForward)
                volumeBox._max = volumeBox._min + maximal;
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

            // TODO: look at different handling orders e.g. slice from middle of container followed
            // by slice from side
            auto slices = makeIndexSequence(_partitioningForward.size());
            std::mutex sliceLock;

            std::vector<std::thread> schedulers(0);

            auto scheduler = [this](const DataContainer<data_t>& x, DataContainer<data_t>& Ax,
                                    std::vector<int>& slices, std::mutex& sliceLock,
                                    std::size_t i) {
                IndexVector_t volumeChunkSize =
                    (_partitioningForward[0].first._max - _partitioningForward[0].first._min)
                        .template cast<index_t>();

                IndexVector_t sinoChunkSize =
                    (_partitioningForward[0].second._max - _partitioningForward[0].second._min)
                        .template cast<index_t>();
                auto cudaVars = _projectors[i]->setupCUDAVariablesForwardConstrained(
                    volumeChunkSize, sinoChunkSize);

                int sliceNum;
                while (true) {
                    // select next slice
                    {
                        std::scoped_lock lock(sliceLock);

                        if (slices.empty())
                            return;

                        sliceNum = slices[slices.size() - 1];
                        slices.pop_back();
                    }

                    const auto& boxes = _partitioningForward[sliceNum];

                    _projectors[i]->applyConstrained(x, Ax, boxes.first, boxes.second, *cudaVars);
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

        std::vector<std::pair<BoundingBox, BoundingBox>> _partitioningForward;

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
    };
} // namespace elsa