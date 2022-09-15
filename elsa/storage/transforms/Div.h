#pragma once

#include "Functions.hpp"
#include "TypeTraits.hpp"

#include "thrust/transform.h"

namespace elsa
{
    namespace detail
    {
        template <class T>
        struct DivVectorScalar {
            DivVectorScalar(T scalar) : scalar_(scalar) {}

            template <class data_t>
            __host__ __device__ auto operator()(const data_t& x) -> std::common_type_t<T, data_t>
            {
                return x / scalar_;
            }

            T scalar_;
        };

        template <class T>
        struct DivScalarVector {
            DivScalarVector(T scalar) : scalar_(scalar) {}

            template <class data_t>
            __host__ __device__ auto operator()(const data_t& x) -> std::common_type_t<T, data_t>
            {
                return scalar_ / x;
            }

            T scalar_;
        };
    } // namespace detail

    /// @brief Compute the component wise division of two vectors
    /// @ingroup transforms
    template <class InputIter1, class InputIter2, class OutIter>
    void div(InputIter1 xfirst, InputIter1 xlast, InputIter2 yfirst, OutIter out)
    {
        thrust::transform(xfirst, xlast, yfirst, out, elsa::divides{});
    }

    /// @brief Compute the component wise division of a vectors and a scalar
    /// @ingroup transforms
    template <class data_t, class InputIter, class OutIter>
    void divScalar(InputIter first, InputIter last, const data_t& scalar, OutIter out)
    {
        // TODO: Find out why a lambda doesn't work here!
        thrust::transform(first, last, out, detail::DivVectorScalar(scalar));
    }

    /// @brief Compute the component wise division of a scalar and a vector
    /// @ingroup transforms
    template <class data_t, class InputIter, class OutIter>
    void divScalar(const data_t& scalar, InputIter first, InputIter last, OutIter out)
    {
        // TODO: Find out why a lambda doesn't work here!
        thrust::transform(first, last, out, detail::DivScalarVector(scalar));
    }
} // namespace elsa
