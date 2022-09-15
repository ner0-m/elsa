#pragma once

#include "transforms/Mul.h"

namespace elsa
{
    /// @brief Multiply the two ranges together, while the first is the output range at the same
    /// time
    /// @ingroup transforms
    template <class InOutIter, class InputIter>
    void inplaceMul(InOutIter xfirst, InOutIter xlast, InputIter yfirst)
    {
        elsa::mul(xfirst, xlast, yfirst, xfirst);
    }

    /// @brief Multiply a range to a scalar, while the given range is also the output range
    /// @ingroup transforms
    template <class InOutIter, class Scalar>
    void inplaceMulScalar(InOutIter xfirst, InOutIter xlast, const Scalar& scalar)
    {
        elsa::mulScalar(xfirst, xlast, scalar, xfirst);
    }
} // namespace elsa
