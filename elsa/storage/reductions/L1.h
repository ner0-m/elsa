#pragma once

#include "TypeTraits.hpp"
#include "Functions.hpp"
#include "functions/Abs.hpp"

#include <thrust/transform_reduce.h>
#include <thrust/iterator/iterator_traits.h>

namespace elsa
{
    /// @brief Compute the l1 norm, i.e. the sum of absolute values, of the vector
    ///
    /// @ingroup reductions
    template <class InputIter>
    auto l1Norm(InputIter first, InputIter last) ->
        typename value_type_of<thrust::iterator_value_t<InputIter>>::type
    {
        using data_t = value_type_of_t<thrust::iterator_value_t<InputIter>>;
        return thrust::transform_reduce(first, last, elsa::abs, data_t(0), elsa::plus{});
    }
} // namespace elsa
