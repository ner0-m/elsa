#pragma once

#include "LinearOperator.h"
#include "BoundingBox.h"
#include "VolumeDescriptor.h"
#include "DetectorDescriptor.h"
#include "CUDAWrappers.h"

namespace elsa
{
    /// wrapper for CUDA variables required by traversal kernel
    struct CUDAVariablesTraversal {
        CudaStreamWrapper stream;

        virtual ~CUDAVariablesTraversal() = default;
    };

    /// wrapper for CUDA variables required by forward traversal kernel
    struct CUDAVariablesForward : public CUDAVariablesTraversal {
        ~CUDAVariablesForward() override = default;
    };

    /// wrapper for CUDA variables required by adjoint traversal kernel
    struct CUDAVariablesAdjoint : public CUDAVariablesTraversal {
        ~CUDAVariablesAdjoint() override = default;
    };

    template <typename data_t>
    class CUDAProjector : public LinearOperator<data_t>
    {
    public:
        /**
         * \brief Determine which part of the volume is responsible for the data measured in the
         * specified part of the image
         *
         * \param[in] startCoordinate the start coordinate of the image part
         * \param[in] endCoordinate the end coordinate of the image part
         *
         * \returns the AABB of the reponsible part of the volume
         */
        virtual BoundingBox
            constrainProjectionSpace(const BoundingBox& sinogramBoundingBox) const = 0;

        // TODO does not check the boxes' bounds -> probably best to make this protected
        virtual void applyConstrained(const DataContainer<data_t>& x, DataContainer<data_t>& Ax,
                                      const BoundingBox& volumeBoundingBox,
                                      const BoundingBox& sinogramBoundingBox,
                                      CUDAVariablesForward& cudaVars) const = 0;

        static inline std::mutex deviceLock = std::mutex();

        /// create and allocate memory for all CUDA variables required by forward traversal kernel
        /// arguments should specify the sizes of the largest chunks which will be traversed
        virtual std::unique_ptr<CUDAVariablesForward>
            setupCUDAVariablesForwardConstrained(IndexVector_t chunkSizeDomain,
                                                 IndexVector_t chunkSizeRange) const = 0;

    protected:
        CUDAProjector(const VolumeDescriptor& domainDescriptor,
                      const DetectorDescriptor& rangeDescriptor, int device = 0)
            : LinearOperator<data_t>{domainDescriptor, rangeDescriptor},
              _volumeDescriptor{domainDescriptor},
              _detectorDescriptor{rangeDescriptor},
              _device{device}
        {
            int deviceCount;
            cudaGetDeviceCount(&deviceCount);

            if (_device >= deviceCount)
                throw std::invalid_argument("CUDAProjector: Tried to select device number "
                                            + std::to_string(_device) + " but only "
                                            + std::to_string(deviceCount) + " devices available");
        }

        /// default copy constructor, hidden from non-derived classes to prevent potential
        /// slicing
        CUDAProjector(const CUDAProjector<data_t>& other)
            : LinearOperator<data_t>{other._volumeDescriptor, other._detectorDescriptor},
              _volumeDescriptor{other._volumeDescriptor},
              _detectorDescriptor{other._detectorDescriptor},
              _device{other._device}
        {
        }

        /// create and allocate memory for all CUDA variables required by forward traversal kernel
        /// arguments should specify the sizes of the largest chunks which will be traversed
        virtual std::unique_ptr<CUDAVariablesForward>
            setupCUDAVariablesForward(IndexVector_t chunkSizeDomain,
                                      IndexVector_t chunkSizeRange) const = 0;

        void containerChunkToPinned(const data_t* dcData, PinnedArray<data_t>& pinned,
                                    const DataDescriptor& dcDesc,
                                    const BoundingBox& chunkBoundingBox) const
        {
            containerChunkPinnedMemoryTransfer<PAGEABLE_TO_PINNED>(
                const_cast<data_t*>(dcData), dcDesc, pinned.get(), chunkBoundingBox);
        }

        void pinnedToContainerChunk(data_t* dcData, const DataDescriptor& dcDesc,
                                    const PinnedArray<data_t>& pinned,
                                    const BoundingBox& chunkBoundingBox) const
        {
            containerChunkPinnedMemoryTransfer<PINNED_TO_PAGEABLE>(
                dcData, dcDesc, const_cast<data_t*>(pinned.get()), chunkBoundingBox);
        }

        /// convenience typedef for the Eigen::Matrix data vector
        using DataVector_t = Eigen::Matrix<data_t, Eigen::Dynamic, 1>;

        /// Reference to VolumeDescriptor stored in LinearOperator
        const VolumeDescriptor& _volumeDescriptor;

        /// Reference to DetectorDescriptor stored in LinearOperator
        const DetectorDescriptor& _detectorDescriptor;

        const int _device;

    private:
        enum MemcpyDirection { PINNED_TO_PAGEABLE, PAGEABLE_TO_PINNED };

        template <MemcpyDirection dir>
        void containerChunkPinnedMemoryTransfer(data_t* dcData, const DataDescriptor& dcDesc,
                                                data_t* pinnedChunk,
                                                const BoundingBox& chunkBoundingBox) const
        {
            auto dcSize = dcDesc.getNumberOfCoefficientsPerDimension();

            auto minSlice = (chunkBoundingBox._min.array().template cast<index_t>() >= 0)
                                .select(chunkBoundingBox._min.template cast<index_t>(), 0);
            auto maxSlice =
                (chunkBoundingBox._max.array().template cast<index_t>() <= dcSize.array())
                    .select(chunkBoundingBox._max.template cast<index_t>(), dcSize);

            IndexVector_t chunkSize =
                (chunkBoundingBox._max - chunkBoundingBox._min).template cast<index_t>();
            IndexVector_t dcChunkSize = (maxSlice - minSlice);

            auto chunkStartIdx = dcDesc.getIndexFromCoordinate(minSlice);
            auto dcColumnSize = dcSize[0];
            auto dcSliceSize = dcSize[1] * dcColumnSize;

            data_t* src = dir == PINNED_TO_PAGEABLE ? pinnedChunk : dcData + chunkStartIdx;
            data_t* dst = dir == PAGEABLE_TO_PINNED ? pinnedChunk : dcData + chunkStartIdx;
            auto srcColumnSize = dir == PINNED_TO_PAGEABLE ? chunkSize[0] : dcColumnSize;
            auto srcSliceSize =
                dir == PINNED_TO_PAGEABLE ? chunkSize[0] * chunkSize[1] : dcSliceSize;
            auto dstColumnSize = dir == PAGEABLE_TO_PINNED ? chunkSize[0] : dcColumnSize;
            auto dstSliceSize =
                dir == PAGEABLE_TO_PINNED ? chunkSize[0] * chunkSize[1] : dcSliceSize;

            // assumes 3d container
            for (index_t z = 0; z < dcChunkSize[2]; z++) {
                for (index_t y = 0; y < dcChunkSize[1]; y++) {
                    // copy chunk to pinned memory column by column
                    std::memcpy((void*) (dst + z * dstSliceSize + y * dstColumnSize),
                                (const void*) (src + z * srcSliceSize + y * srcColumnSize),
                                dcChunkSize[0] * sizeof(data_t));
                }
            }
        }

        /// lift from base class
        using LinearOperator<data_t>::_domainDescriptor;
        using LinearOperator<data_t>::_rangeDescriptor;
    };
} // namespace elsa