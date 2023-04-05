/**
 * @file TraverseJosephsCUDA.cuh
 *
 * @brief Provides interface definitions for the Joseph's CUDA projector. Allows for separable
 * compilation of device and host code.
 *
 * @author Nikola Dinev (nikola.dinev@tum.de)
 */
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "elsaDefines.h"

namespace elsa
{

    template <typename data_t = real_t, uint dim = 3>
    struct TraverseJosephsCUDA {
        constexpr static uint32_t MAX_THREADS_PER_BLOCK = 32;

        /**
         *  Allows for the bounding box to be passed to the kernel by value.
         *  Kernel arguments are stored in constant memory, and should generally
         *  provide faster access to the variables than via global memory.
         */
        struct BoundingBox {
            // min is always 0

            real_t _max[dim];
            __device__ __forceinline__ const real_t& operator[](const uint32_t idx) const
            {
                return _max[idx];
            }
            __device__ __forceinline__ real_t& operator[](const uint32_t idx) { return _max[idx]; }
        };

        /**
         * @brief Forward projection using Josephs's method
         *
         * @param[in] sinogramDims specifies the dimensions of the sinogram
         * @param[in] threads specifies the number of threads for each block
         * @param[in] volume handle to the texture object containing the volume data
         * @param[out] sinogram pointer to output
         * @param[in] sinogramPitch pitch (padded width in bytes) of sinogram
         * @param[in] rayOrigins pointer to ray origins
         * @param[in] originPitch pitch of ray origins
         * @param[in] projInv pointer to inverse of projection matrices (stored column-wise)
         * @param[in] projPitch pitch of inverse of projection matrices
         * @param[in] boundingBox specifies the size of the volume
         *
         * sinogramDims should be given as (numAngles, detectorSizeY (1 for 2D), detectorSizeX).
         *
         * threads should ideally be a multiple of the warp size (32 for all current GPUs).
         *
         * @warning Interpolation mode for texture should be set to cudaFilterModePoint when working
         * with doubles.
         */
        static void traverseForward(dim3 sinogramDims, int threads, cudaTextureObject_t volume,
                                    int8_t* __restrict__ sinogram, uint64_t sinogramPitch,
                                    const int8_t* __restrict__ rayOrigins, uint32_t originPitch,
                                    const int8_t* __restrict__ projInv, uint32_t projPitch,
                                    const BoundingBox& boundingBox);

        /**
         * @brief Backward projection using Josephs's method
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
         *
         * This is the matched version of the forward traversal. Considerably slower
         * as interpolation is performed in software.
         */
        static void traverseAdjoint(dim3 sinogramDims, int threads, int8_t* __restrict__ volume,
                                    uint64_t volumePitch, const int8_t* __restrict__ sinogram,
                                    uint64_t sinogramPitch, const int8_t* __restrict__ rayOrigins,
                                    uint32_t originPitch, const int8_t* __restrict__ projInv,
                                    uint32_t projPitch, const BoundingBox& boundingBox);

        /**
         * @brief Approximation of backward projection for Josephs's method
         *
         * @param[in] volumeDims specifies the dimensions of the volume
         * @param[in] threads specifies the number of threads for each block
         * @param[out] volume pointer to output
         * @param[in] volumePitch pitch (padded width in bytes) of volume
         * @param[in] sinogram handle to the texture object containing the volume data
         * @param[in] rayOrigins pointer to ray origins
         * @param[in] originPitch pitch of ray origins
         * @param[in] proj pointer to projection matrices
         * @param[in] projPitch pitch of projection matrices
         * @param[in] numAngles number of acquisition angles
         *
         * volumeDims should be given as (volSizeZ (1 for 2D), volSizeY, volSizeX).
         *
         * threads should ideally be a multiple of the warp size (32 for all current GPUs).
         *
         * @warning Interpolation mode for texture should be set to cudaFilterModePoint when working
         * with doubles.
         */
        static void traverseAdjointFast(dim3 volumeDims, int threads, int8_t* __restrict__ volume,
                                        uint64_t volumePitch, cudaTextureObject_t sinogram,
                                        const int8_t* __restrict__ rayOrigins, uint32_t originPitch,
                                        const int8_t* __restrict__ proj, uint32_t projPitch,
                                        uint32_t numAngles);
    };
} // namespace elsa