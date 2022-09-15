#pragma once

#include "TypeTraits.hpp"
#include "functions/Real.hpp"
#include "transforms/Cast.h"

#include "thrust/transform.h"
#include "thrust/copy.h"
#include "thrust/iterator/iterator_traits.h"

namespace elsa
{
    /// @brief Extract the real part of a range. If the input range is not complex, it is equivalent
    /// to a copy.
    ///
    /// @ingroup transforms
    template <class InputIter, class OutIter>
    void real(InputIter first, InputIter last, OutIter out)
    {
        using indata_t = thrust::iterator_value_t<InputIter>;

        // If the input is not complex, the imaginary part will be 0
        if constexpr (!is_complex_v<indata_t>) {
            thrust::copy(first, last, out);
        } else {
            thrust::transform(first, last, out, elsa::fn::real);
        }
    }
} // namespace elsa
