#pragma once

#include "SharedArray.cuh"

template <typename real_t>
__device__ __forceinline__ real_t initDelta(real_t rd, int stepDir)
{
    real_t d = stepDir / rd;
    return rd >= -__FLT_EPSILON__ && rd <= __FLT_EPSILON__ ? __FLT_MAX__ : d;
}

/// initialize step sizes considering the ray direcion
template <typename real_t, uint32_t dim, uint32_t max_threads>
__device__ __forceinline__ void
    initDelta(const elsa::detail::EasyAccessSharedArray<real_t, dim, max_threads> rd,
              elsa::detail::EasyAccessSharedArray<real_t, dim, max_threads> delta)
{
#pragma unroll
    for (uint32_t i = 0; i < dim; i++) {
        int stepDir = (rd[i] > 0.0f) - (rd[i] < 0.0f);
        delta[i] = initDelta(rd[i], stepDir);
    }
}

template <typename real_t, uint32_t dim, uint32_t max_threads>
__device__ __forceinline__ void
    initDelta(const real_t* const __restrict__ rd,
              const elsa::detail::EasyAccessSharedArray<int, dim, max_threads>& stepDir,
              elsa::detail::EasyAccessSharedArray<real_t, dim, max_threads>& delta)
{
#pragma unroll
    for (int i = 0; i < dim; i++) {
        delta[i] = initDelta(rd[i], stepDir[i]);
    }
}

/// returns the index of the smallest element in an array
template <typename real_t, uint32_t dim, uint32_t max_threads>
__device__ __forceinline__ uint32_t
    minIndex(const elsa::detail::EasyAccessSharedArray<real_t, dim, max_threads>& array)
{
    uint32_t index = 0;
    real_t min = array[0];

#pragma unroll
    for (uint32_t i = 1; i < dim; i++) {
        bool cond = array[i] < min;
        index = cond ? i : index;
        min = cond ? array[i] : min;
    }

    return index;
}

/// return the index of the element with the maximum absolute value in array
template <typename real_t, uint32_t dim, uint32_t max_threads>
__device__ __forceinline__ uint32_t
    maxAbsIndex(const elsa::detail::EasyAccessSharedArray<real_t, dim, max_threads>& array)
{
    uint32_t index = 0;
    real_t max = fabs(array[0]);

#pragma unroll
    for (uint32_t i = 1; i < dim; i++) {
        bool cond = fabs(array[i]) > max;
        index = cond ? i : index;
        max = cond ? fabs(array[i]) : max;
    }

    return index;
}
