#pragma once

#include "functions/Abs.hpp"

#include <thrust/transform.h>

namespace elsa
{
    /// @brief Compute the coefficient wise absolute value of the input ranges.
    /// @ingroup transforms
    template <class InputIter, class OutIter>
    void cwiseAbs(InputIter first, InputIter last, OutIter out)
    {
        thrust::transform(first, last, out, elsa::abs);
    }
} // namespace elsa
