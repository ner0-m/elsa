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
              _projector(volumeDescriptor, detectorDescriptor),
              _targetDevices(targetDevices),
              _partitioningForward{}
        {
            int deviceCount = numAvailableDevices();
            index_t sliceThickness = 8;

            for (const int& targetDevice : targetDevices)
                if (targetDevice >= deviceCount)
                    throw std::invalid_argument("SplittingProjector: Tried to select device number "
                                                + std::to_string(targetDevice) + " but only "
                                                + std::to_string(deviceCount)
                                                + " devices available");

            const auto detectorSize = detectorDescriptor.getNumberOfCoefficientsPerDimension();
            for (index_t i = 0; i < detectorSize[1]; i += sliceThickness) {
                IndexVector_t startCoordinate(3);
                startCoordinate << 0, i, 0;
                IndexVector_t endCoordinate = detectorSize;
                if (i + sliceThickness <= detectorSize[1])
                    endCoordinate[1] = i + sliceThickness;

                const BoundingBox sinoBox(startCoordinate, endCoordinate);
                const auto volumeBox = _projector.constrainProjectionSpace(sinoBox);

                _partitioningForward.emplace_back(volumeBox, sinoBox);
            }
        }

    protected:
        /// apply forward projection
        void applyImpl(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const override
        {
            Timer timeguard("SplittingProjector", "apply");

            // TODO: look at different handling orders e.g. slice from middle of container followed
            // by slice from side
            auto slices = makeIndexSequence(_partitioningForward.size());
            std::mutex sliceLock;

            std::vector<std::thread> schedulers(0);
            for (const auto& i : _targetDevices)
                schedulers.emplace_back(
                    [this](const DataContainer<data_t>& x, DataContainer<data_t>& Ax,
                           std::vector<int>& slices, std::mutex& sliceLock, int device) {
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

                            _projector.applyConstrained(x, Ax, boxes.first, boxes.second, device);
                        }
                    },
                    std::cref(x), std::ref(Ax), std ::ref(slices), std::ref(sliceLock), i);

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
        ProjectionMethod _projector;

        std::vector<int> _targetDevices;

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

        void sliceScheduler(const DataContainer<data_t>& x, DataContainer<data_t>& Ax,
                            std::vector<int>& slices, std::mutex& sliceLock, int device)
        {
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

                _projector.applyConstrained(x, Ax, boxes.first, boxes.second, device);
            }
        }
    };
} // namespace elsa