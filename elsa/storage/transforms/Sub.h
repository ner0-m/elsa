#pragma once

#include "Functions.hpp"

#include <thrust/transform.h>

namespace elsa
{
    namespace detail
    {
        template <class T>
        struct SubVectorScalar {
            SubVectorScalar(T scalar) : scalar_(scalar) {}

            template <class data_t>
            __host__ __device__ auto operator()(const data_t& x) -> std::common_type_t<T, data_t>
            {
                using U = std::common_type_t<T, data_t>;
                return static_cast<U>(x) - static_cast<U>(scalar_);
            }

            T scalar_;
        };

        template <class T>
        struct SubScalarVector {
            SubScalarVector(T scalar) : scalar_(scalar) {}

            template <class data_t>
            __host__ __device__ auto operator()(const data_t& x) -> std::common_type_t<T, data_t>
            {
                using U = std::common_type_t<T, data_t>;
                return static_cast<U>(scalar_) - static_cast<U>(x);
            }

            T scalar_;
        };
    } // namespace detail

    /// @brief Compute the component wise subtraction of two vectors
    /// @ingroup transforms
    template <class InputIter1, class InputIter2, class OutIter>
    void sub(InputIter1 xfirst, InputIter1 xlast, InputIter2 yfirst, OutIter out)
    {
        thrust::transform(xfirst, xlast, yfirst, out, elsa::minus{});
    }

    /// @brief Compute the component wise subtraction of a vectors and a scalar
    /// @ingroup transforms
    template <class data_t, class InputIter, class OutIter>
    void subScalar(InputIter first, InputIter last, const data_t& scalar, OutIter out)
    {
        // TODO: Find out why a lambda doesn't work here!
        thrust::transform(first, last, out, detail::SubVectorScalar(scalar));
    }

    /// @brief Compute the component wise subtraction of a scalar and a vector
    /// @ingroup transforms
    template <class data_t, class InputIter, class OutIter>
    void subScalar(const data_t& scalar, InputIter first, InputIter last, OutIter out)
    {
        // TODO: Find out why a lambda doesn't work here!
        thrust::transform(first, last, out, detail::SubScalarVector(scalar));
    }
} // namespace elsa
