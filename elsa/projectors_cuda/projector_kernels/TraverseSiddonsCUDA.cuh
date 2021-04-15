/**
 * @file TraverseSiddonsCUDA.cuh
 *
 * @brief Provides interface definitions for the Siddon's CUDA projector. Allows for separable
 * compilation of device and host code.
 *
 * @author Nikola Dinev (nikola.dinev@tum.de)
 */
#pragma once

#include "cuda_runtime.h"
#include <cstdint>
#include "elsaDefines.h"

namespace elsa
{

    template <typename data_t = real_t, uint32_t dim = 3>
    struct TraverseSiddonsCUDA {

        const static uint32_t MAX_THREADS_PER_BLOCK = 64;
        /**
         *  Allows for the bounding box to be passed to the kernel by value.
         *  Kernel arguments are stored in constant memory, and should generally
         *  provide faster access to the variables than via global memory.
         */
        struct BoundingBox {
            // min is always 0

            uint32_t _max[dim];
            __device__ __forceinline__ const uint32_t& operator[](const uint32_t idx) const
            {
                return _max[idx];
            }
            __device__ __forceinline__ uint32_t& operator[](const uint32_t idx)
            {
                return _max[idx];
            }
        };

        /**
         * @brief Forward projection using Siddon's method
         *
         * @param[in] sinogramDims specifies the dimensions of the sinogram
         * @param[in] threads specifies the number of threads for each block
         * @param[in] volume pointer to volume data
         * @param[in] volumePitch pitch (padded width in bytes) of volume
         * @param[out] sinogram pointer to output
         * @param[in] sinogramPitch pitch of sinogram
         * @param[in] rayOrigins pointer to ray origins
         * @param[in] originPitch pitch of ray origins
         * @param[in] projInv pointer to inverse of projection matrices
         * @param[in] projPitch pitch of inverse of projection matrices
         * @param[in] boundingBox specifies the size of the volume
         *
         * sinogramDims should be given as (numAngles, detectorSizeY (1 for 2D), detectorSizeX).
         *
         * threads should ideally be a multiple of the warp size (32 for all current GPUs).
         */
        static void traverseForward(dim3 sinogramDims, int threads, int8_t* __restrict__ volume,
                                    uint64_t volumePitch, int8_t* __restrict__ sinogram,
                                    uint64_t sinogramPitch, const int8_t* __restrict__ rayOrigins,
                                    uint32_t originPitch, const int8_t* __restrict__ projInv,
                                    uint32_t projPitch, const BoundingBox& boundingBox);

        /**
         * @brief Backward projection using Siddon's method
         *
         * @param[in] sinogramDims specifies the dimensions of the sinogram
         * @param[in] threads specifies the number of threads for each block
         * @param[out] volume pointer to output
         * @param[in] volumePitch pitch (padded width in bytes) of volume
         * @param[in] sinogram pointer to sinogram data
         * @param[in] sinogramPitch pitch of sinogram
         * @param[in] rayOrigins pointer to ray origins
         * @param[in] originPitch pitch of ray origins
         * @param[in] projInv pointer to inverse of projection matrices
         * @param[in] projPitch pitch of inverse of projection matrices
         * @param[in] boundingBox specifies the size of the volume
         *
         * sinogramDims should be given as (numAngles, detectorSizeY (1 for 2D), detectorSizeX).
         *
         * threads should ideally be a multiple of the warp size (32 for all current GPUs).
         */
        static void traverseAdjoint(dim3 blocks, int threads, int8_t* __restrict__ volume,
                                    uint64_t volumePitch, int8_t* __restrict__ sinogram,
                                    uint64_t sinogramPitch, const int8_t* __restrict__ rayOrigins,
                                    uint32_t originPitch, const int8_t* __restrict__ projInv,
                                    uint32_t projPitch, const BoundingBox& boundingBox);
    };
} // namespace elsa