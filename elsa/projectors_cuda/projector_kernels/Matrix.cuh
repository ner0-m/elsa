#pragma once

#include <cuda_runtime.h>
#include "SharedArray.cuh"

/**
 * @brief General square matrix-vector multiplication
 *
 * important: always use byte pointers for multidimensional arrays
 */
template <typename real_t, uint32_t dim, typename Array>
__device__ __forceinline__ void gesqmv(const int8_t* const __restrict__ matrix,
                                       const real_t* const __restrict__ vector, Array result,
                                       const uint32_t matrixPitch)
{
    // initialize result vector
    real_t* columnPtr = (real_t*) matrix;
#pragma unroll
    for (uint32_t x = 0; x < dim; x++) {
        result[x] = columnPtr[x] * vector[0];
    }

// accumulate results for remaning columns
#pragma unroll
    for (uint32_t y = 1; y < dim; y++) {
        real_t* columnPtr = (real_t*) (matrix + matrixPitch * y);
#pragma unroll
        for (uint32_t x = 0; x < dim; x++) {
            result[x] += columnPtr[x] * vector[y];
        }
    }
}

/// determine reverse norm of vector of length 2 or 3 using device inbuilt functions
template <typename real_t, uint32_t dim, typename Array>
__device__ __forceinline__ real_t rnorm(Array vector)
{
    if (dim == 3)
        return rnorm3d(vector[0], vector[1], vector[2]);
    else if (dim == 2)
        return rhypot(vector[0], vector[1]);
    else {
        real_t acc = vector[0];
#pragma unroll
        for (uint32_t i = 1; i < dim; i++)
            acc += vector[i];

        return acc;
    }
}

template <typename real_t, uint32_t dim, uint32_t max_threads>
__device__ __forceinline__ void
    normalize(elsa::detail::EasyAccessSharedArray<real_t, dim, max_threads>& vector)
{
    real_t rn = rnorm<real_t, dim>(vector);

#pragma unroll
    for (uint32_t i = 0; i < dim; i++) {
        vector[i] *= rn;
    }
}

template <typename real_t, uint32_t dim>
__device__ __forceinline__ void normalize(real_t* vector)
{
    real_t rn = rnorm<real_t, dim>(vector);

#pragma unroll
    for (uint32_t i = 0; i < dim; i++) {
        vector[i] *= rn;
    }
}
