#pragma once

#include "transforms/Sub.h"

namespace elsa
{
    /// @brief Subtract the two ranges together, while the first is the output range at the same
    /// time
    /// @ingroup transforms
    template <class InOutIter, class InputIter>
    void inplaceSub(InOutIter xfirst, InOutIter xlast, InputIter yfirst)
    {
        elsa::sub(xfirst, xlast, yfirst, xfirst);
    }

    /// @brief Add a scalar from a range, while the given range is also the output range
    /// @ingroup transforms
    template <class InOutIter, class Scalar>
    void inplaceSubScalar(InOutIter xfirst, InOutIter xlast, const Scalar& scalar)
    {
        elsa::subScalar(xfirst, xlast, scalar, xfirst);
    }
} // namespace elsa
