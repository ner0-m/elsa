#pragma once

#include <cuda_runtime.h>

namespace elsa::detail
{
    template <typename data_t, uint32_t size, uint32_t MAX_THREADS_PER_BLOCK>
    struct EasyAccessSharedArray {
        data_t* const __restrict__ _p;

        __device__ EasyAccessSharedArray(data_t* p) : _p{p + threadIdx.x} {}

        __device__ __forceinline__ const data_t& operator[](uint32_t index) const
        {
            return _p[index * MAX_THREADS_PER_BLOCK];
        }

        __device__ __forceinline__ data_t& operator[](uint32_t index)
        {
            return _p[index * MAX_THREADS_PER_BLOCK];
        }
    };
} // namespace elsa::detail
