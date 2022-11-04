#pragma once

#include "Functions.hpp"

#include <thrust/transform_reduce.h>

namespace elsa
{
    /// @brief Compute the sum of the vector (\f$\sum_i x_i\f$.)
    ///
    /// @ingroup reductions
    template <class InputIter>
    auto sum(InputIter first, InputIter last) -> thrust::iterator_value_t<InputIter>
    {
        using data_t = thrust::iterator_value_t<InputIter>;

        return thrust::transform_reduce(
            first, last, [] __host__ __device__(const data_t& val) { return val; }, data_t(0),
            thrust::plus{});
    }
} // namespace elsa
