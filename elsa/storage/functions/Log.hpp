#pragma once

#include "CUDADefines.h"

#include <thrust/complex.h>

#include <cmath>
#include <complex>

namespace elsa::fn
{
    namespace detail
    {
        struct LogFn {
            template <class T>
            __host__ __device__ constexpr T operator()(const T& arg) const noexcept
            {
                using thrust::log;
                using std::log;

                return log(arg);
            }
        };
    } // namespace detail

    static constexpr __device__ detail::LogFn log;
} // namespace elsa::fn
