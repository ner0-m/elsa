#pragma once

#include "CUDADefines.h"
#include <thrust/complex.h>
#include <complex>

namespace elsa::fn
{
    namespace detail
    {
        struct sign_fn {
            template <class T>
            __host__ constexpr auto operator()(const T& arg) const noexcept
            {
                return (T{0} < arg) - (arg < T{0});
            }

            template <class T>
            __host__ __device__ constexpr auto
                operator()(const thrust::complex<T>& arg) const noexcept -> T
            {
                if (arg.real() > 0) {
                    return T{1};
                } else if (arg.real() < 0) {
                    return T{-1};
                } else {
                    return (*this)(arg.imag());
                }
            }
        };
    } // namespace detail

    static constexpr __device__ detail::sign_fn sign;
} // namespace elsa::fn
