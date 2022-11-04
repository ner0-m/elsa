#pragma once

#include "CUDADefines.h"

#include <thrust/complex.h>

#include <cmath>
#include <complex>

namespace elsa::fn
{
    namespace detail
    {
        struct ExpFn {
            template <class T>
            __host__ __device__ constexpr T operator()(const T& arg) const noexcept
            {
                using thrust::exp;
                using std::exp;

                return exp(arg);
            }
        };
    } // namespace detail

    static constexpr __device__ detail::ExpFn exp;
} // namespace elsa::fn
