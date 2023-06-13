#pragma once

#include "functions/Bessel.hpp"
#include "Bessel.h"

#include <thrust/transform.h>

namespace elsa
{
    /// @brief Compute the log of modified Bessel function of the first kind
    /// of order zero for each element of the input range
    /// @ingroup transforms
    template <class InputIter, class OutIter>
    void bessel_log_0(InputIter first, InputIter last, OutIter out)
    {
        thrust::transform(first, last, out, elsa::fn::bessel_log_0);
    }

    /// @brief Compute the modified Bessel function of the first kind
    /// of order one divided by that of order zero for each element of
    /// the input range
    /// @ingroup transforms
    template <class InputIter, class OutIter>
    void bessel_1_0(InputIter first, InputIter last, OutIter out)
    {
        thrust::transform(first, last, out, elsa::fn::bessel_1_0);
    }
} // namespace elsa
