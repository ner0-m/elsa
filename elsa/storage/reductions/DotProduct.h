#pragma once

#include "TypeTraits.hpp"
#include "functions/Conj.hpp"
#include "Functions.hpp"
#include "../CublasTransforms.h"

#include <thrust/complex.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/iterator_traits.h>

namespace elsa
{
    /// @brief Compute the dot product between two vectors
    ///
    /// Compute the sum of products of each entry in the vectors, i.e. \f$\sum_i x_i * y_i\f$.
    /// If any of the two vectors is complex, the dot product is conjugate linear in the first
    /// component and linear in the second, i.e. \f$\sum_i \bar{x}_i * y_i\f$, as is done in Eigen
    /// and Numpy.
    ///
    /// The return type is determined from the value types of the two iterators. If any is a complex
    /// type, the return type will also be a complex type.
    ///
    /// @ingroup reductions
    template <class InputIter1, class InputIter2,
              class data_t = std::common_type_t<thrust::iterator_value_t<InputIter1>,
                                                thrust::iterator_value_t<InputIter2>>>
    auto dot(InputIter1 xfirst, InputIter1 xlast, InputIter2 yfirst)
        -> std::common_type_t<thrust::iterator_value_t<InputIter1>,
                              thrust::iterator_value_t<InputIter2>>
    {
        using xdata_t = thrust::iterator_value_t<InputIter1>;
        using ydata_t = thrust::iterator_value_t<InputIter2>;
        using common_t = std::common_type_t<thrust::iterator_value_t<InputIter1>,
                              thrust::iterator_value_t<InputIter2>>;

        // using data_t = std::common_type_t<xdata_t, ydata_t>;
        common_t temp = common_t();
        if (cublas::inplaceDotProduct<InputIter1, InputIter2, common_t>(xfirst, xlast, yfirst, temp))
            return temp;

        if constexpr (is_specialization_v<
                          xdata_t,
                          thrust::complex> || is_specialization_v<ydata_t, thrust::complex>) {
            return thrust::inner_product(
                xfirst, xlast, yfirst, data_t(0), elsa::plus{},
                [] __host__ __device__(const xdata_t& x, const ydata_t& y) {
                    return elsa::fn::conj(x) * y;
                });
        } else {
            return thrust::inner_product(xfirst, xlast, yfirst, xdata_t(0));
        }
    }
} // namespace elsa
