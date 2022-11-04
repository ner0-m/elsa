#pragma once

#include "Functions.hpp"

#include <thrust/transform.h>

namespace elsa
{
    namespace detail
    {
        template <class T>
        struct MulVectorScalar {
            MulVectorScalar(T scalar) : scalar_(scalar) {}

            template <class data_t>
            __host__ __device__ auto operator()(const data_t& x) -> std::common_type_t<T, data_t>
            {
                using U = std::common_type_t<T, data_t>;
                return static_cast<U>(x) * static_cast<U>(scalar_);
            }

            T scalar_;
        };

        template <class T>
        struct MulScalarVector {
            MulScalarVector(T scalar) : scalar_(scalar) {}

            template <class data_t>
            __host__ __device__ auto operator()(const data_t& x) -> std::common_type_t<T, data_t>
            {
                using U = std::common_type_t<T, data_t>;
                return static_cast<U>(scalar_) * static_cast<U>(x);
            }

            T scalar_;
        };
    } // namespace detail

    /// @brief Compute the component wise multiplication of two vectors
    /// @ingroup transforms
    template <class InputIter1, class InputIter2, class OutIter>
    void mul(InputIter1 xfirst, InputIter1 xlast, InputIter2 yfirst, OutIter out)
    {
        thrust::transform(xfirst, xlast, yfirst, out, elsa::multiplies{});
    }

    /// @brief Compute the component wise multiplies of a vectors and a scalar
    /// @ingroup transforms
    template <class data_t, class InputIter, class OutIter>
    void mulScalar(InputIter first, InputIter last, const data_t& scalar, OutIter out)
    {
        // TODO: Find out why a lambda doesn't work here!
        thrust::transform(first, last, out, detail::MulVectorScalar(scalar));
    }

    /// @brief Compute the component wise multiplication of a scalar and a vector
    /// @ingroup transforms
    template <class data_t, class InputIter, class OutIter>
    void mulScalar(const data_t& scalar, InputIter first, InputIter last, OutIter out)
    {
        // TODO: Find out why a lambda doesn't work here!
        thrust::transform(first, last, out, detail::MulScalarVector(scalar));
    }
} // namespace elsa
