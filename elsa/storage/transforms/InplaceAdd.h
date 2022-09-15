#pragma once

#include "TypeTraits.hpp"
#include "transforms/Add.h"

namespace elsa
{
    /// @brief Add the two ranges together, while the first is the output range at the same time
    /// @ingroup transforms
    template <class InOutIter, class InputIter>
    void inplaceAdd(InOutIter xfirst, InOutIter xlast, InputIter yfirst)
    {
        elsa::add(xfirst, xlast, yfirst, xfirst);
    }

    /// @brief Add a range to a scalar, while the given range is also the output range
    /// @ingroup transforms
    template <class InOutIter, class Scalar>
    void inplaceAddScalar(InOutIter xfirst, InOutIter xlast, const Scalar& scalar)
    {
        elsa::addScalar(xfirst, xlast, scalar, xfirst);
    }
} // namespace elsa
