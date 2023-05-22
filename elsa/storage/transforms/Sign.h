#pragma once

#include "functions/Sign.hpp"

#include <thrust/transform.h>

namespace elsa
{
    /// @brief apply the log function for each element in the range
    /// @ingroup transforms
    template <class InputIter, class OutIter>
    void sign(InputIter first, InputIter last, OutIter out)
    {
        thrust::transform(first, last, out, elsa::fn::sign);
    }
} // namespace elsa
