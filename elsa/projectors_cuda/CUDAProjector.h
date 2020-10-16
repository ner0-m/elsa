#pragma once

#include "LinearOperator.h"
#include "BoundingBox.h"
#include "VolumeDescriptor.h"
#include "DetectorDescriptor.h"
#include "elsaDefinesCUDA.cuh"
#include "Logger.h"

#include <cuda_runtime.h>

#include <memory>

/**
 * \brief Custom macro to check CUDA API calls for errors with line information
 */
#ifndef gpuErrchk
#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        if (abort) {
            elsa::Logger::get("CUDAProjector")
                ->critical("GPUassert: {} {} {}", cudaGetErrorString(code), file, line);
            exit(code);
        } else {
            elsa::Logger::get("CUDAProjector")
                ->error("GPUassert: {} {} {}", cudaGetErrorString(code), file, line);
        }
    }
}
#endif

namespace elsa
{
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
                                      const BoundingBox& sinogramBoundingBox, int device) const = 0;

        static inline std::mutex deviceLock = std::mutex();

    protected:
        CUDAProjector(const VolumeDescriptor& domainDescriptor,
                      const DetectorDescriptor& rangeDescriptor)
            : LinearOperator<data_t>{domainDescriptor, rangeDescriptor},
              _volumeDescriptor{static_cast<const VolumeDescriptor&>(*_domainDescriptor)},
              _detectorDescriptor{static_cast<const DetectorDescriptor&>(*_rangeDescriptor)}
        {
        }

        /// default copy constructor, hidden from non-derived classes to prevent potential
        /// slicing
        CUDAProjector(const CUDAProjector<data_t>&) = default;

        PinnedArray<data_t> containerChunkToPinned(const data_t* dcData,
                                                   const DataDescriptor& dcDesc,
                                                   const BoundingBox& chunkBoundingBox) const
        {
            PinnedArray<data_t> chunk(
                (chunkBoundingBox._max - chunkBoundingBox._min).template cast<index_t>().prod());

            containerChunkPinnedMemoryTransfer<PAGEABLE_TO_PINNED>(
                const_cast<data_t*>(dcData), dcDesc, chunk.get(), chunkBoundingBox);

            return chunk;
        }

        void pinnedToContainerChunk(data_t* dcData, const DataDescriptor& dcDesc,
                                    const data_t* pinned, const BoundingBox& chunkBoundingBox) const
        {
            containerChunkPinnedMemoryTransfer<PINNED_TO_PAGEABLE>(
                dcData, dcDesc, const_cast<data_t*>(pinned), chunkBoundingBox);
        }

        /// convenience typedef for the Eigen::Matrix data vector
        using DataVector_t = Eigen::Matrix<data_t, Eigen::Dynamic, 1>;

        /// Reference to VolumeDescriptor stored in LinearOperator
        const VolumeDescriptor& _volumeDescriptor;

        /// Reference to DetectorDescriptor stored in LinearOperator
        const DetectorDescriptor& _detectorDescriptor;

    private:
        enum MemcpyDirection { PINNED_TO_PAGEABLE, PAGEABLE_TO_PINNED };

        template <MemcpyDirection dir>
        void containerChunkPinnedMemoryTransfer(data_t* dcData, const DataDescriptor& dcDesc,
                                                data_t* pinnedChunk,
                                                const BoundingBox& chunkBoundingBox) const
        {
            IndexVector_t chunkSize =
                (chunkBoundingBox._max - chunkBoundingBox._min).template cast<index_t>();

            IndexVector_t chunkStart = chunkBoundingBox._min.template cast<index_t>();

            auto chunkStartIdx = dcDesc.getIndexFromCoordinate(chunkStart);
            auto dcColumnSize = dcDesc.getNumberOfCoefficientsPerDimension()[0];
            auto dcSliceSize = dcDesc.getNumberOfCoefficientsPerDimension()[1] * dcColumnSize;

            data_t* src = dir == PINNED_TO_PAGEABLE ? pinnedChunk : dcData + chunkStartIdx;
            data_t* dst = dir == PAGEABLE_TO_PINNED ? pinnedChunk : dcData + chunkStartIdx;
            auto srcColumnSize = dir == PINNED_TO_PAGEABLE ? chunkSize[0] : dcColumnSize;
            auto srcSliceSize =
                dir == PINNED_TO_PAGEABLE ? chunkSize[0] * chunkSize[1] : dcSliceSize;
            auto dstColumnSize = dir == PAGEABLE_TO_PINNED ? chunkSize[0] : dcColumnSize;
            auto dstSliceSize =
                dir == PAGEABLE_TO_PINNED ? chunkSize[0] * chunkSize[1] : dcSliceSize;

            // assumes 3d container
            for (index_t z = 0; z < chunkSize[2]; z++) {
                for (index_t y = 0; y < chunkSize[1]; y++) {
                    // copy chunk to pinned memory column by column
                    std::memcpy((void*) (dst + z * dstSliceSize + y * dstColumnSize),
                                (const void*) (src + z * srcSliceSize + y * srcColumnSize),
                                chunkSize[0] * sizeof(data_t));
                }
            }
        }

        /// lift from base class
        using LinearOperator<data_t>::_domainDescriptor;
        using LinearOperator<data_t>::_rangeDescriptor;
    };
} // namespace elsa