#pragma once

#include "CUDADefines.h"

namespace elsa::fn
{
    namespace detail
    {
        struct SquareFn {
            template <class T>
            __host__ __device__ constexpr T operator()(const T& arg) const noexcept
            {
                return arg * arg;
            }
        };
    } // namespace detail

    static constexpr __device__ detail::SquareFn square;
} // namespace elsa::fn
