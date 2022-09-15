#pragma once

#include "Functions.hpp"
#include "TypeTraits.hpp"
#include "functions/Abs.hpp"
#include "functions/Square.hpp"

#include "thrust/transform_reduce.h"

namespace elsa
{
    /// @brief Compute the squared L2-norm, the sum of squares (\f$\sum_i x_i * x_i\f$.)
    ///
    /// @ingroup reductions
    template <class InputIter>
    auto squaredL2Norm(InputIter first, InputIter last)
        -> value_type_of_t<thrust::iterator_value_t<InputIter>>
    {
        using data_t = thrust::iterator_value_t<InputIter>;
        using inner_t = value_type_of_t<data_t>;

        if constexpr (is_complex_v<data_t>) {
            return thrust::transform_reduce(
                first, last,
                [] __host__ __device__(const data_t& val) {
                    return elsa::fn::square(elsa::abs(val));
                },
                inner_t(0), elsa::plus{});
        } else {
            return thrust::transform_reduce(
                first, last,
                [] __host__ __device__(const data_t& val) { return elsa::fn::square(val); },
                data_t(0), elsa::plus{});
        }
    }

    /// @brief Compute the L2-norm, the square root of the sum of squares (\f$\sqrt{\sum_i x_i *
    /// x_i}\f$.)
    ///
    /// @ingroup reductions
    template <class InputIter>
    auto l2Norm(InputIter first, InputIter last)
        -> value_type_of_t<thrust::iterator_value_t<InputIter>>
    {
        using std::sqrt;
        using thrust::sqrt;

        return sqrt(elsa::squaredL2Norm(first, last));
    }
} // namespace elsa
