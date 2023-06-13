#pragma once

#include <thrust/transform.h>

namespace elsa
{
    namespace detail
    {
        template <class data_t>
        struct LincombFunctor {
            data_t a_;
            data_t b_;

            LincombFunctor(data_t a, data_t b) : a_(a), b_(b) {}

            __host__ __device__ data_t operator()(const data_t& x, const data_t& y) const
            {
                return a_ * x + b_ * y;
            }
        };
    } // namespace detail

    /// @brief Compute the linear combination of \f$a * x + b * y\f$, where
    /// \f$x\f$ and \f$y\f$ are vectors given as iterators, and written to the output
    /// iterator
    ///
    /// @ingroup transforms
    template <class data_t, class InputIter1, class InputIter2, class OutIter>
    void lincomb(data_t a, InputIter1 first1, InputIter1 last1, data_t b, InputIter2 first2,
                 OutIter out)
    {
        thrust::transform(first1, last1, first2, out, detail::LincombFunctor<data_t>{a, b});
    }
} // namespace elsa
