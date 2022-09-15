#pragma once

#include "CUDADefines.h"
#include "thrust/complex.h"
#include <complex>

namespace elsa::fn
{
    namespace detail
    {
        struct conj_fn {
            template <class T>
            __host__ __device__ constexpr auto
                operator()(const thrust::complex<T>& arg) const noexcept
                -> decltype(thrust::conj(std::declval<thrust::complex<T>>()))
            {
                return thrust::conj(arg);
            }

            template <class T>
            constexpr auto operator()(const std::complex<T>& arg) const noexcept
                -> decltype(std::conj(std::declval<std::complex<T>>()))
            {
                return std::conj(arg);
            }
        };
    } // namespace detail

    static constexpr __device__ detail::conj_fn conj;
} // namespace elsa::fn
