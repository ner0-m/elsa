#pragma once

#include "CUDADefines.h"

#include "thrust/complex.h"

#include <cmath>
#include <complex>

namespace elsa::fn
{
    namespace detail
    {
        struct SqrtFn {
            template <class T>
            __host__ __device__ constexpr T operator()(const T& arg) const noexcept
            {
                using std::sqrt;
                using thrust::sqrt;

                return sqrt(arg);
            }
        };
    } // namespace detail

    static constexpr __device__ detail::SqrtFn sqrt;
} // namespace elsa::fn
