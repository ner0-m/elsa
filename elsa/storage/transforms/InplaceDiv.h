#pragma once

#include "TypeTraits.hpp"
#include "transforms/Div.h"

namespace elsa
{
    /// @brief Divide the two ranges coefficient wise, while the first is the output range at the
    /// same time
    /// @ingroup transforms
    template <class InOutIter, class InputIter>
    void inplaceDiv(InOutIter xfirst, InOutIter xlast, InputIter yfirst)
    {
        elsa::div(xfirst, xlast, yfirst, xfirst);
    }

    /// @brief Divide the range coefficient wise with a scalar, while the first is the output
    /// range at the same time
    /// @ingroup transforms
    template <class InOutIter, class Scalar>
    void inplaceDivScalar(InOutIter xfirst, InOutIter xlast, Scalar scalar)
    {
        elsa::divScalar(xfirst, xlast, scalar, xfirst);
    }
} // namespace elsa
