#pragma once

#include "thrust/transform.h"
#include "thrust/iterator/iterator_traits.h"

namespace elsa
{
    /// @brief Cast input range to type from output range
    /// @ingroup transforms
    template <class InputIter, class OutIter>
    void cast(InputIter first, InputIter last, OutIter out)
    {
        using From = thrust::remove_cvref_t<thrust::iterator_value_t<InputIter>>;
        using To = thrust::remove_cvref_t<thrust::iterator_value_t<OutIter>>;

        thrust::transform(first, last, out,
                          [] __host__ __device__(const From& val) { return To(val); });
    }
} // namespace elsa
