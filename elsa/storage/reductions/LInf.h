#pragma once

#include "TypeTraits.hpp"
#include "functions/Abs.hpp"
#include "DisableWarnings.h"

DISABLE_WARNING_PUSH
DISABLE_WARNING_SIGN_CONVERSION
#include <thrust/iterator/transform_iterator.h>
#include <thrust/extrema.h>
DISABLE_WARNING_POP

namespace elsa
{
    /// @brief Compute the max of the vector (\f$\sup_i |x_i|\f$.)
    ///
    /// @ingroup reductions
    template <class InputIter>
    auto lInf(InputIter first, InputIter last)
        -> value_type_of_t<thrust::iterator_value_t<InputIter>>
    {
        using data_t = value_type_of_t<thrust::iterator_value_t<InputIter>>;

        auto zipfirst = thrust::make_transform_iterator(first, elsa::abs);
        auto ziplast = thrust::make_transform_iterator(last, elsa::abs);

        auto iter = thrust::max_element(zipfirst, ziplast);

        return iter == ziplast ? data_t() : *iter;
    }
} // namespace elsa
