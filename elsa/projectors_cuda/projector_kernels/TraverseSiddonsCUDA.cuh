/**
 * \file TraverseSiddonsCUDA.cuh
 * 
 * \brief Provides interface definitions for the Siddon's CUDA projector. Allows for separable compilation of device and host code.
 * 
 * \author Nikola Dinev (nikola.dinev@tum.de)
 */
#pragma once

#include "cuda_runtime.h"
#include "stdint.h"
#include "elsaDefines.h"

namespace elsa {

    template <typename data_t = real_t, uint32_t dim = 3>
    struct TraverseSiddonsCUDA {

        const static uint32_t MAX_THREADS_PER_BLOCK = 64;
        /**
         *  Allows for the bounding box to be passed to the kernel by value.
         *  Kernel arguments are stored in constant memory, and should generally 
         *  provide faster access to the variables than via global memory.
         */ 
        struct BoundingBox
        {
            //min is always 0

            uint32_t max[dim];
            __device__ __forceinline__ const uint32_t &operator[](const uint32_t idx) const
            {
                return max[idx];
            }
            __device__ __forceinline__ uint32_t &operator[](const uint32_t idx)
            {
                return max[idx];
            }
        };

        /**
         * \brief Forward projection using Siddon's method
         * 
         * \param[in] blocks specifies the grid used for kernel execution
         * \param[in] threads specifies the number of threads for each block
         * \param[in] volume pointer to volume data
         * \param[in] volumePitch pitch (padded width in bytes) of volume
         * \param[out] sinogram pointer to output
         * \param[in] sinogramPitch pitch of sinogram
         * \param[in] rayOrigins pointer to ray origins
         * \param[in] originPitch pitch of ray origins
         * \param[in] projInv pointer to inverse of projection matrices
         * \param[in] projPitch pitch of inverse of projection matrices
         * \param[in] boxMax specifies the size of the volume
         * \param[in] stream handle to stream in which the kernel should be placed
         * 
         * The variables blocks and threads should be picked based on the sinogram dimensions. To process all 
         * rays set blocks to (detectorSizeX, detectorSizeY, numAngles / threads), if numAngles is not a multiple of threads
         * a second kernel call must be made to process the remaining rays with blocks = (detectorSizeX, detectorSizeY, 1)
         * and threads = numAngles % threadsFirstCall. Sinogram, projection matrix, and ray origin pointers should be
         * adjusted accordingly to point to the start of the (numAngles - numAngles % threadsFirstCall)-th element.
         * 
         * threads should ideally be a multiple of the warp size (32 for all current GPUs).
         */ 
        static void traverseForward(const dim3 blocks,
            const int threads,
            int8_t* const __restrict__ volume,
            const uint64_t volumePitch,
            int8_t* const __restrict__ sinogram,
            const uint64_t sinogramPitch,
            const int8_t* const __restrict__ rayOrigins,
            const uint32_t originPitch,
            const int8_t* const __restrict__ projInv,
            const uint32_t projPitch,
            const BoundingBox& boxMax,
            cudaStream_t stream = (cudaStream_t)0);
        
        /**
         * \brief Backward projection using Siddon's method
         * 
         * \param[in] blocks specifies the grid used for kernel execution
         * \param[in] threads specifies the number of threads for each block
         * \param[out] volume pointer to output
         * \param[in] volumePitch pitch (padded width in bytes) of volume
         * \param[in] sinogram pointer to sinogram data
         * \param[in] sinogramPitch pitch of sinogram
         * \param[in] rayOrigins pointer to ray origins
         * \param[in] originPitch pitch of ray origins
         * \param[in] projInv pointer to inverse of projection matrices
         * \param[in] projPitch pitch of inverse of projection matrices
         * \param[in] boxMax specifies the size of the volume
         * \param[in] stream handle to stream in which the kernel should be placed
         * 
         * The variables blocks and threads should be picked based on the sinogram dimensions. To process all 
         * rays set blocks to (detectorSizeX, detectorSizeY, numAngles / threads), if numAngles is not a multiple of threads
         * a second kernel call must be made to process the remaining rays with blocks = (detectorSizeX, detectorSizeY, 1)
         * and threads = numAngles % threadsFirstCall. Sinogram, projection matrix, and ray origin pointers should be
         * adjusted accordingly to point to the start of the (numAngles - numAngles % threadsFirstCall)-th element.
         * 
         * threads should ideally be a multiple of the warp size (32 for all current GPUs).
         */ 
        static void traverseAdjoint(const dim3 blocks,
            const int threads,
            int8_t* const __restrict__ volume,
            const uint64_t volumePitch,
            int8_t* const __restrict__ sinogram,
            const uint64_t sinogramPitch,
            const int8_t* const __restrict__ rayOrigins,
            const uint32_t originPitch,
            const int8_t* const __restrict__ projInv,
            const uint32_t projPitch,
            const BoundingBox& boxMax,
            cudaStream_t stream = (cudaStream_t)0);
    };
}