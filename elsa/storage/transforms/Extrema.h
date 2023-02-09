#pragma once

#include "functions/Abs.hpp"

#include <thrust/transform.h>
#include <thrust/extrema.h>
#include <type_traits>

namespace elsa
{
    namespace detail
    {
        struct CwiseMaxFn {
            template <class T, class U>
            __host__ __device__ auto operator()(const T& lhs, const U& rhs) const noexcept
                -> std::common_type_t<T, U>
            {
                using data_t = std::common_type_t<T, U>;
                return thrust::max(static_cast<data_t>(lhs), static_cast<data_t>(rhs));
            }

            template <class T, class U>
            __host__ __device__ auto operator()(const thrust::complex<T>& lhs,
                                                const U& rhs) const noexcept
                -> std::common_type_t<T, U>
            {
                using data_t = std::common_type_t<T, U>;
                return thrust::max(static_cast<data_t>(elsa::abs(lhs)), static_cast<data_t>(rhs));
            }

            template <class T, class U>
            __host__ __device__ auto operator()(const T& lhs,
                                                const thrust::complex<U>& rhs) const noexcept
                -> std::common_type_t<T, U>
            {
                using data_t = std::common_type_t<T, U>;
                return thrust::max(static_cast<data_t>(lhs), static_cast<data_t>(elsa::abs(rhs)));
            }

            template <class T, class U>
            __host__ __device__ auto operator()(const thrust::complex<T>& lhs,
                                                const thrust::complex<U>& rhs) const noexcept
                -> std::common_type_t<T, U>
            {
                using data_t = std::common_type_t<T, U>;
                return thrust::max(static_cast<data_t>(elsa::abs(lhs)),
                                   static_cast<data_t>(elsa::abs(rhs)));
            }
        };

        struct CwiseMinFn {
            template <class T, class U>
            __host__ __device__ auto operator()(const T& lhs, const U& rhs) const noexcept
                -> std::common_type_t<T, U>
            {
                using data_t = std::common_type_t<T, U>;
                return thrust::min(static_cast<data_t>(lhs), static_cast<data_t>(rhs));
            }

            template <class T, class U>
            __host__ __device__ auto operator()(const thrust::complex<T>& lhs,
                                                const U& rhs) const noexcept
                -> std::common_type_t<T, U>
            {
                using data_t = std::common_type_t<T, U>;
                return thrust::min(static_cast<data_t>(elsa::abs(lhs)), static_cast<data_t>(rhs));
            }

            template <class T, class U>
            __host__ __device__ auto operator()(const T& lhs,
                                                const thrust::complex<U>& rhs) const noexcept
                -> std::common_type_t<T, U>
            {
                using data_t = std::common_type_t<T, U>;
                return thrust::min<data_t>(lhs, elsa::abs(rhs));
            }

            template <class T, class U>
            __host__ __device__ auto operator()(const thrust::complex<T>& lhs,
                                                const thrust::complex<U>& rhs) const noexcept
                -> std::common_type_t<T, U>
            {
                using data_t = std::common_type_t<T, U>;
                return thrust::min<data_t>(elsa::abs(lhs), elsa::abs(rhs));
            }
        };

        struct TernaryLess {
            template <class T, class U>
            __host__ __device__ auto operator()(const T& lhs, const U& rhs) const noexcept
                -> std::common_type_t<T, U>
            {
                using data_t = std::common_type_t<T, U>;
                return thrust::less{}(lhs, rhs) ? static_cast<data_t>(lhs)
                                                : static_cast<data_t>(rhs);
            }

            template <class T, class U>
            __host__ __device__ auto operator()(const thrust::complex<T>& lhs,
                                                const thrust::complex<U>& rhs) const noexcept
                -> thrust::complex<std::common_type_t<T, U>>
            {
                using data_t = std::common_type_t<T, U>;
                return thrust::less{}(static_cast<data_t>(elsa::abs(lhs)),
                                      static_cast<data_t>(elsa::abs(rhs)))
                           ? lhs
                           : rhs;
            }
        };

        struct TernaryGreater {
            template <class T, class U>
            __host__ __device__ auto operator()(const T& lhs, const U& rhs) const noexcept
                -> std::common_type_t<T, U>
            {
                using data_t = std::common_type_t<T, U>;
                return thrust::greater{}(lhs, rhs) ? static_cast<data_t>(lhs)
                                                   : static_cast<data_t>(rhs);
            }

            template <class T, class U>
            __host__ __device__ auto operator()(const thrust::complex<T>& lhs,
                                                const thrust::complex<U>& rhs) const noexcept
                -> thrust::complex<std::common_type_t<T, U>>
            {
                using data_t = std::common_type_t<T, U>;
                return thrust::greater{}(static_cast<data_t>(elsa::abs(lhs)),
                                         static_cast<data_t>(elsa::abs(rhs)))
                           ? lhs
                           : rhs;
            }
        };

        template <class Scalar>
        struct CwiseMinScalarFn {
            Scalar scalar_;

            template <class T>
            __host__ __device__ auto operator()(const T& val) const noexcept
            {
                return TernaryGreater{}(val, scalar_);
            }
        };

        template <class Scalar>
        struct CwiseMaxScalarFn {
            Scalar scalar_;

            template <class T>
            __host__ __device__ auto operator()(const T& val) const noexcept
            {
                return TernaryLess{}(val, scalar_);
            }
        };

    } // namespace detail

    /// @brief Compute the coefficient wise maximum of two input ranges. For complex input's the
    /// absolute value of the complex number is used.
    /// @ingroup transforms
    template <class InputIter1, class InputIter2, class OutIter>
    void cwiseMax(InputIter1 xfirst, InputIter1 xlast, InputIter2 yfirst, OutIter out)
    {
        thrust::transform(xfirst, xlast, yfirst, out, detail::CwiseMaxFn{});
    }

    /// @brief Compute the coefficient wise minimum of two input ranges. For complex input's the
    /// absolute value of the complex number is used.
    /// @ingroup transforms
    template <class InputIter1, class InputIter2, class OutIter>
    void cwiseMin(InputIter1 xfirst, InputIter1 xlast, InputIter2 yfirst, OutIter out)
    {
        thrust::transform(xfirst, xlast, yfirst, out, detail::CwiseMinFn{});
    }

    /// @brief For each element in the vector set the element to the minimum of the element and the
    /// given scalar
    /// @ingroup transforms
    template <class InputIter1, class Scalar, class OutIter>
    void minimum(InputIter1 xfirst, InputIter1 xlast, Scalar scalar, OutIter out)
    {
        thrust::transform(xfirst, xlast, out, detail::CwiseMinScalarFn<Scalar>{scalar});
    }

    /// @brief For each element in the vector set the element to the maximum of the element and the
    /// given scalar
    /// @ingroup transforms
    template <class InputIter1, class Scalar, class OutIter>
    void maximum(InputIter1 xfirst, InputIter1 xlast, Scalar scalar, OutIter out)
    {
        thrust::transform(xfirst, xlast, out, detail::CwiseMaxScalarFn<Scalar>{scalar});
    }
} // namespace elsa
