#pragma once

#include <cuda_runtime.h>
#include "SharedArray.cuh"

/**
 * @brief General square matrix-vector multiplication
 *
 * important: always use byte pointers for multidimensional arrays
 */
template <typename data_t, uint32_t dim, typename Array>
__device__ __forceinline__ void gesqmv(const int8_t* const __restrict__ matrix,
                                       const data_t* const __restrict__ vector, Array result,
                                       const uint32_t matrixPitch)
{
    // initialize result vector
    data_t* columnPtr = (data_t*) matrix;
#pragma unroll
    for (uint32_t x = 0; x < dim; x++) {
        result[x] = columnPtr[x] * vector[0];
    }

// accumulate results for remaning columns
#pragma unroll
    for (uint32_t y = 1; y < dim; y++) {
        data_t* columnPtr = (data_t*) (matrix + matrixPitch * y);
#pragma unroll
        for (uint32_t x = 0; x < dim; x++) {
            result[x] += columnPtr[x] * vector[y];
        }
    }
}

/**
 * @brief General homogenous matrix-vector multiplication
 *
 * a matrix that has shape (dim) x (dim + 1)
 * important: always use byte pointers for multidimensional arrays
 * @param dim: real dim without homogenous
 */
template <uint32_t dim, typename data_t, typename Array>
__device__ __forceinline__ void gehomv(const data_t* const __restrict__ matrix,
                                       const data_t* const __restrict__ vector, Array result)
{
    // initialize result vector
    const data_t* columnPtr = matrix;
#pragma unroll
    for (uint32_t x = 0; x < dim; x++) {
        result[x] = columnPtr[x] * vector[0];
    }

// accumulate results for remaning columns
#pragma unroll
    for (uint32_t y = 1; y < dim + 1; y++) {
        columnPtr = matrix + dim * y;
#pragma unroll
        for (uint32_t x = 0; x < dim; x++) {
            result[x] += columnPtr[x] * vector[y];
        }
    }
}

/// determine norm of vector of length 1, 2 or 3 using device inbuilt functions
template <uint32_t dim>
__device__ __forceinline__ float norm(float* vector)
{
    return normf(dim, vector);
}

template <uint32_t dim>
__device__ __forceinline__ double norm(double* vector)
{
    return norm(dim, vector);
}

/// determine reverse norm of vector of length 2 or 3 using device inbuilt functions
template <typename data_t, uint32_t dim, typename Array>
__device__ __forceinline__ data_t rnorm(Array vector)
{
    if (dim == 3)
        return rnorm3d(vector[0], vector[1], vector[2]);
    else if (dim == 2)
        return rhypot(vector[0], vector[1]);
    else {
        data_t acc = vector[0];
#pragma unroll
        for (uint32_t i = 1; i < dim; i++)
            acc += vector[i];

        return acc;
    }
}

template <typename data_t, uint32_t dim, uint32_t max_threads>
__device__ __forceinline__ void
    normalize(elsa::detail::EasyAccessSharedArray<data_t, dim, max_threads>& vector)
{
    data_t rn = rnorm<data_t, dim>(vector);

#pragma unroll
    for (uint32_t i = 0; i < dim; i++) {
        vector[i] *= rn;
    }
}

template <typename data_t, uint32_t dim>
__device__ __forceinline__ void normalize(data_t* vector)
{
    data_t rn = rnorm<data_t, dim>(vector);

#pragma unroll
    for (uint32_t i = 0; i < dim; i++) {
        vector[i] *= rn;
    }
}

template <typename data_t, uint32_t dim>
__device__ __forceinline__ void homogenousNormalize(data_t* vector)
{
    data_t w = 1 / vector[dim - 1];

#pragma unrolls
    for (uint32_t i = 0; i < dim - 1; i++) {
        vector[i] *= w;
    }
}