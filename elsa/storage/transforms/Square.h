#pragma once

#include "functions/Square.hpp"

#include "thrust/transform.h"

namespace elsa
{
    /// @brief Compute the square for each element of the input range
    /// @ingroup transforms
    template <class InputIter, class OutIter>
    void square(InputIter first, InputIter last, OutIter out)
    {
        thrust::transform(first, last, out, elsa::fn::square);
    }
} // namespace elsa
