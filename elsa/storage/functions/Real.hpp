#pragma once

#include "CUDADefines.h"
#include "TypeTraits.hpp"
#include "thrust/complex.h"
#include <complex>

namespace elsa::fn
{
    namespace detail
    {
        struct real_fn {
            template <class T>
            __host__ __device__ constexpr auto operator()(const T& arg) const noexcept
                -> std::enable_if_t<!is_complex_v<T>, T>
            {
                return arg;
            }

            template <class T>
            __host__ __device__ constexpr auto
                operator()(const thrust::complex<T>& arg) const noexcept
                -> decltype(std::declval<thrust::complex<T>>().real())
            {
                return arg.real();
            }

            template <class T>
            __host__ constexpr auto operator()(const std::complex<T>& arg) const noexcept
                -> decltype(std::real(std::declval<std::complex<T>>()))
            {
                return std::real(arg);
            }
        };
    } // namespace detail

    static constexpr __device__ detail::real_fn real;
} // namespace elsa::fn
