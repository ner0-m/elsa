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

template <typename data_t = float, uint dim = 3>
    struct TraverseJosephsCUDA
    {
        const static uint32_t MAX_THREADS_PER_BLOCK = 64;

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

        static void traverseForward(const dim3 blocks, const int threads,
                                    cudaTextureObject_t volume,
                                    int8_t *const __restrict__ sinogram,
                                    const uint64_t sinogramPitch,
                                    const int8_t *const __restrict__ rayOrigins,
                                    const uint32_t originPitch,
                                    const int8_t *const __restrict__ projInv,
                                    const uint32_t projPitch,
                                    const BoundingBox boxMax,
                                    const cudaStream_t stream = 0);

        /**
             * \brief Acts as the exact adjoint of the forward traversal operator.
             * 
             * Volume has to be zero initialized to guarantee correct output.
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
                                    BoundingBox boxMax,
                                    const cudaStream_t stream = 0);

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