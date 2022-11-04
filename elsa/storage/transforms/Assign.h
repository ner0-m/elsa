#pragma once

#include <thrust/fill.h>
#include <thrust/copy.h>
#include <type_traits>

namespace elsa
{
    /// @brief Fill given range with scalar value
    /// @ingroup transforms
    template <class InputIter, class data_t>
    void fill(InputIter first, InputIter last, const data_t& scalar)
    {
        thrust::fill(first, last, scalar);
    }

    /// @brief Copy input range to the output range
    /// @ingroup transforms
    template <class InputIter, class OutputIter>
    void assign(InputIter first, InputIter last, OutputIter out)
    {
        thrust::copy(first, last, out);
    }
} // namespace elsa
