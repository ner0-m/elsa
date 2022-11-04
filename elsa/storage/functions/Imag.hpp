#pragma once

#include "CUDADefines.h"
#include "TypeTraits.hpp"
#include <thrust/complex.h>
#include <complex>

namespace elsa::fn
{
    namespace detail
    {
        struct imag_fn {
            template <class T>
            __host__ __device__ constexpr auto operator()(const T&) const noexcept
                -> std::enable_if_t<!is_complex_v<T>, T>
            {
                return 0;
            }

            template <class T>
            __host__ __device__ constexpr auto
                operator()(const thrust::complex<T>& arg) const noexcept
                -> decltype(std::declval<thrust::complex<T>>().imag())
            {
                return arg.imag();
            }

            template <class T>
            __host__ constexpr auto operator()(const std::complex<T>& arg) const noexcept
                -> decltype(std::imag(std::declval<std::complex<T>>()))
            {
                return std::imag(arg);
            }
        };
    } // namespace detail

    static constexpr __device__ detail::imag_fn imag;
} // namespace elsa::fn
