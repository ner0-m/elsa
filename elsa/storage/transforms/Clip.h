#pragma once

#include "ContiguousStorage.h"

#include "thrust/transform.h"
#include "thrust/iterator/iterator_traits.h"

namespace elsa
{
    /// @brief Clip input range to `minval` and `maxval`
    /// @ingroup transforms
    template <class Iter, class OutIter, class T = thrust::iterator_value_t<Iter>,
              class U = thrust::iterator_value_t<Iter>>
    void clip(Iter first, Iter last, OutIter out, const T& minval, const U& maxval)
    {
        using data_t = thrust::iterator_value_t<Iter>;

        thrust::transform(first, last, out, [=] __host__ __device__(const data_t& x) {
            if (x < static_cast<data_t>(minval)) {
                return static_cast<data_t>(minval);
            } else if (x > static_cast<data_t>(maxval)) {
                return static_cast<data_t>(maxval);
            } else {
                return x;
            }
        });
    }

    /// @brief Clip input range to `0` and `maxval`
    /// @ingroup transforms
    template <class Iter, class OutIter, class T = thrust::iterator_value_t<Iter>>
    void clip(Iter first, Iter last, OutIter out, const T& maxval)
    {
        using data_t = thrust::iterator_value_t<Iter>;
        return clip(first, last, out, data_t(0), static_cast<data_t>(maxval));
    }
} // namespace elsa
