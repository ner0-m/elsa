#pragma once

#include "CUDADefines.h"
#include "thrust/complex.h"
#include <complex>

namespace elsa
{
    namespace detail
    {
        struct abs_fn {
            template <class T>
            __host__ constexpr auto operator()(const T& arg) const noexcept
                -> decltype(std::abs(std::declval<T>()))
            {
                return std::abs(arg);
            }

            template <class T>
            __host__ __device__ constexpr auto
                operator()(const thrust::complex<T>& arg) const noexcept
                -> decltype(thrust::abs(std::declval<thrust::complex<T>>()))
            {
                return thrust::abs(arg);
            }
        };
    } // namespace detail

    static constexpr __device__ detail::abs_fn abs;
} // namespace elsa
