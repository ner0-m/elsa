#pragma once

#include "TypeTraits.hpp"
#include "Functions.hpp"

#include "thrust/transform.h"

namespace elsa
{
    namespace detail
    {
        template <class T>
        struct AddVectorScalar {
            AddVectorScalar(T scalar) : scalar_(scalar) {}

            template <class data_t>
            __host__ __device__ auto operator()(const data_t& x) -> std::common_type_t<T, data_t>
            {
                return x + scalar_;
            }

            T scalar_;
        };

        template <class T>
        struct AddScalarVector {
            AddScalarVector(T scalar) : scalar_(scalar) {}

            template <class data_t>
            __host__ __device__ auto operator()(const data_t& x) -> std::common_type_t<T, data_t>
            {
                return scalar_ + x;
            }

            T scalar_;
        };
    } // namespace detail

    /// @brief Compute the component wise addition of two vectors
    /// @ingroup transforms
    template <class InputIter1, class InputIter2, class OutIter>
    void add(InputIter1 xfirst, InputIter1 xlast, InputIter2 yfirst, OutIter out)
    {
        thrust::transform(xfirst, xlast, yfirst, out, elsa::plus{});
    }

    /// @brief Compute the component wise addition of a vectors and a scalar
    /// @ingroup transforms
    template <class data_t, class InputIter, class OutIter>
    void addScalar(InputIter first, InputIter last, const data_t& scalar, OutIter out)
    {
        // TODO: Find out why a lambda doesn't work here!
        thrust::transform(first, last, out, detail::AddVectorScalar(scalar));
    }

    /// @brief Compute the component wise addition of a scalar and a vector
    /// @ingroup transforms
    template <class data_t, class InputIter, class OutIter>
    void addScalar(const data_t& scalar, InputIter first, InputIter last, OutIter out)
    {
        // TODO: Find out why a lambda doesn't work here!
        thrust::transform(first, last, out, detail::AddScalarVector(scalar));
    }
} // namespace elsa
