#pragma once

#include "functions/Sqrt.hpp"

#include "thrust/transform.h"

namespace elsa
{
    /// @brief Compute the square root for each element of the input range
    /// @ingroup transforms
    template <class InputIter, class OutIter>
    void sqrt(InputIter first, InputIter last, OutIter out)
    {
        thrust::transform(first, last, out, elsa::fn::sqrt);
    }
} // namespace elsa
