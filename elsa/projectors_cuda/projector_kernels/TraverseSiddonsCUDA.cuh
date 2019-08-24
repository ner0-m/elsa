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
#include "elsa.h"

namespace elsa {

    template <typename data_t = real_t, uint32_t dim = 3>
    struct TraverseSiddonsCUDA {
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
            const uint32_t* const __restrict__ boxMax,
            cudaStream_t stream = (cudaStream_t)0);
        
        /**
         * \brief Acts as the exact adjoint of the forward traversal operator.
         * 
         * Volume has to be zero initialized to guarantee correct output.
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
            const uint32_t* const __restrict__ boxMax,
            cudaStream_t stream = (cudaStream_t)0);
    };
}