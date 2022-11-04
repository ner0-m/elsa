#pragma once

#include "TypeTraits.hpp"
#include "functions/Imag.hpp"

#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/iterator/iterator_traits.h>

namespace elsa
{
    /// @brief Extract the imaginary part of a range. If input range is not complex, it is treated
    /// as complex numbers with an imaginary part of `0`
    ///
    /// @ingroup transforms
    template <class InputIter, class OutIter>
    void imag(InputIter first, InputIter last, OutIter out)
    {
        using xdata_t = thrust::iterator_value_t<InputIter>;
        using outdata_t = thrust::iterator_value_t<OutIter>;

        // If the input is not complex, the imaginary part will be 0
        if constexpr (!is_complex_v<xdata_t>) {
            thrust::fill_n(out, thrust::distance(first, last), outdata_t(0));
        } else {
            thrust::transform(first, last, out, elsa::fn::imag);
        }
    }
} // namespace elsa
