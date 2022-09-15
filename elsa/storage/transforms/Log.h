#pragma once

#include "functions/Log.hpp"

#include "thrust/transform.h"

namespace elsa
{
    /// @brief apply the log function for each element in the range
    /// @ingroup transforms
    template <class InputIter, class OutIter>
    void log(InputIter first, InputIter last, OutIter out)
    {
        thrust::transform(first, last, out, elsa::fn::log);
    }
} // namespace elsa
