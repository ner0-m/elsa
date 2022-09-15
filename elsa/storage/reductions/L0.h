#pragma once

#include "functions/Abs.hpp"

#include "thrust/complex.h"
#include "thrust/count.h"
#include "thrust/limits.h"

namespace elsa
{
    namespace detail
    {
        struct nonZeroComp {
            template <class T>
            __host__ __device__ std::ptrdiff_t operator()(const T& arg) const noexcept
            {
                return elsa::abs(arg) >= thrust::numeric_limits<T>::epsilon();
            }

            template <class T>
            __host__ __device__ std::ptrdiff_t
                operator()(const thrust::complex<T>& arg) const noexcept
            {
                return elsa::abs(arg) >= thrust::numeric_limits<T>::epsilon();
            }
        };
    } // namespace detail

    /// @brief compute the l0-"norm", which counts the number of non-zero elements.
    ///
    /// @ingroup reductions
    template <class InputIter>
    std::ptrdiff_t l0PseudoNorm(InputIter first, InputIter last)
    {
        return thrust::count_if(first, last, detail::nonZeroComp{});
    }
} // namespace elsa
