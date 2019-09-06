/**
 * \file TraverseJosephsCUDA.cuh
 * 
 * \brief Provides interface definitions for the Joseph's CUDA projector. Allows for separable compilation of device and host code.
 * 
 * \author Nikola Dinev (nikola.dinev@tum.de)
 */
#pragma once

#include <stdint.h>

#include <cuda_runtime.h>

#include "elsa.h"

namespace elsa
{

template <typename data_t = elsa::real_t, uint dim = 3>
    struct TraverseJosephsCUDA
    {
        const static uint32_t MAX_THREADS_PER_BLOCK = 32;

        /**
         *  Allows for the bounding box to be passed to the kernel by value.
         *  Kernel arguments are stored in constant memory, and should generally 
         *  provide faster access to the variables than via global memory.
         */ 
        struct BoundingBox
        {
            //min is always 0

            real_t max[dim];
            __device__ __forceinline__ const real_t &operator[](const uint32_t idx) const
            {
                return max[idx];
            }
            __device__ __forceinline__ real_t &operator[](const uint32_t idx)
            {
                return max[idx];
            }
        };

        /**
         * \brief Forward projection using Josephs's method
         * 
         * \param[in] blocks specifies the grid used for kernel execution
         * \param[in] threads specifies the number of threads for each block
         * \param[in] volume handle to the texture object containing the volume data
         * \param[out] sinogram pointer to output
         * \param[in] sinogramPitch pitch (padded width in bytes) of sinogram
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
         * 
         * \warning Interpolation mode for texture should be set to cudaFilterModePoint when working with doubles.
         */ 
        static void traverseForward(const dim3 blocks, const int threads,
                                    cudaTextureObject_t volume,
                                    int8_t *const __restrict__ sinogram,
                                    const uint64_t sinogramPitch,
                                    const int8_t *const __restrict__ rayOrigins,
                                    const uint32_t originPitch,
                                    const int8_t *const __restrict__ projInv,
                                    const uint32_t projPitch,
                                    const BoundingBox& boxMax,
                                    const cudaStream_t stream = 0);

        /**
         * \brief Backward projection using Josephs's method
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
         * 
         * This is the matched version of the forward traversal. Considerably slower 
         * as interpolation is performed in software.
         */ 
        static void traverseAdjoint(const dim3 blocks, const int threads,
                                    int8_t *const __restrict__ volume,
                                    const uint64_t volumePitch,
                                    const int8_t *const __restrict__ sinogram,
                                    const uint64_t sinogramPitch,
                                    const int8_t *const __restrict__ rayOrigins,
                                    const uint32_t originPitch,
                                    const int8_t *const __restrict__ projInv,
                                    const uint32_t projPitch,
                                    const BoundingBox& boxMax,
                                    const cudaStream_t stream = 0);

        /**
         * \brief Approximation of backward projection for Josephs's method
         * 
         * \param[in] blocks specifies the grid used for kernel execution
         * \param[in] threads specifies the number of threads for each block
         * \param[out] volume pointer to output
         * \param[in] volumePitch pitch (padded width in bytes) of volume
         * \param[in] sinogram handle to the texture object containing the volume data
         * \param[in] rayOrigins pointer to ray origins
         * \param[in] originPitch pitch of ray origins
         * \param[in] proj pointer to projection matrices
         * \param[in] projPitch pitch of projection matrices
         * \param[in] numAngles number of acquisition angles
         * \param[in] zOffset z-index-offset to part of volume that should be processed
         * \param[in] stream handle to stream in which the kernel should be placed
         * 
         * The variables blocks and threads should be picked based on the volume dimensions. To process all 
         * voxels set blocks to (volSizeX, volSizeY, volSizeZ / threads), if volSizeZ is not a multiple of threads
         * a second kernel call must be made to process the remaining voxels with blocks = (volSizeX, volSizeY, 1)
         * and threads = volSizeZ % threadsFirstCall. Volume pointers should be
         * adjusted accordingly to point to the start of volume[0][0][volSizeZ - volSizeZ % threadsFirstCall].
         * 
         * threads should ideally be a multiple of the warp size (32 for all current GPUs).
         * 
         * \warning Interpolation mode for texture should be set to cudaFilterModePoint when working with doubles.
         */ 
        static void traverseAdjointFast(const dim3 blocks, const int threads,
                                        int8_t *const __restrict__ volume,
                                        const uint64_t volumePitch,
                                        cudaTextureObject_t sinogram,
                                        const int8_t *const __restrict__ rayOrigins,
                                        const uint32_t originPitch,
                                        const int8_t *const __restrict__ proj,
                                        const uint32_t projPitch,
                                        const uint32_t numAngles,
                                        const uint32_t zOffset = 0,
                                        const cudaStream_t stream = 0);
    };
} // namespace elsa