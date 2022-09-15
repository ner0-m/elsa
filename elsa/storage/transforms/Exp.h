#pragma once

#include "functions/Exp.hpp"

#include "thrust/transform.h"

namespace elsa
{
    /// @brief apply the exponentional function for each element in the range
    /// @ingroup transforms
    template <class InputIter, class OutIter>
    void exp(InputIter first, InputIter last, OutIter out)
    {
        thrust::transform(first, last, out, elsa::fn::exp);
    }
} // namespace elsa
