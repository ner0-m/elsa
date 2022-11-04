#pragma once

#include "Functions.hpp"
#include "functions/Abs.hpp"

#include <thrust/extrema.h>
#include <thrust/complex.h>

namespace elsa
{
    namespace detail
    {
        struct MinMaxComp {
            template <class T>
            __host__ __device__ T operator()(const T& lhs, const T& rhs) const noexcept
            {
                return lhs < rhs;
            }

            template <class T>
            __host__ __device__ T operator()(const thrust::complex<T>& lhs,
                                             const thrust::complex<T>& rhs) const noexcept
            {
                return elsa::abs(lhs) < elsa::abs(rhs);
            }
        };
    } // namespace detail

    /// @brief Compute the minimum element of the given vector
    ///
    /// The minimum is determined via the `operator<` of the iterators value type. If the vector is
    /// empty, a default constructed value type is returned.
    ///
    /// @ingroup reductions
    template <class InputIter>
    auto minElement(InputIter first, InputIter last) -> thrust::iterator_value_t<InputIter>
    {
        using data_t = thrust::iterator_value_t<InputIter>;

        auto iter = thrust::min_element(first, last, detail::MinMaxComp{});
        return iter == last ? data_t() : *iter;
    }

    /// @brief Compute the maximum element of the given vector
    ///
    /// The maximum is determined via the `operator<` of the iterators value type. If the vector is
    /// empty, a default constructed value type is returned.
    ///
    /// @ingroup reductions
    template <class InputIter>
    auto maxElement(InputIter first, InputIter last) -> thrust::iterator_value_t<InputIter>
    {
        using data_t = thrust::iterator_value_t<InputIter>;

        auto iter = thrust::max_element(first, last, detail::MinMaxComp{});
        return iter == last ? data_t() : *iter;
    }
} // namespace elsa
