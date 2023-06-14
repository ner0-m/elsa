#include "BSplines.h"

namespace elsa
{
    template <typename data_t>
    BSpline<data_t>::BSpline(index_t dim, index_t order) : dim_(dim), order_(order)
    {
    }

    template <typename data_t>
    data_t BSpline<data_t>::operator()(Vector_t<data_t> x)
    {
        ELSA_VERIFY(dim_ == x.size());
        return bspline::nd_bspline_evaluate(x, order_);
    }

    template <typename data_t>
    data_t BSpline<data_t>::derivative(data_t s)
    {
        return bspline::bsplineDerivative1d_evaluate(s, order_);
    }

    template <typename data_t>
    index_t BSpline<data_t>::order() const
    {
        return order_;
    }

    template <typename data_t>
    data_t BSpline<data_t>::radius() const
    {
        return (order_ + 1) * 0.5;
    }

    template <typename data_t>
    index_t BSpline<data_t>::dim() const
    {
        return dim_;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class BSpline<float>;
    template class BSpline<double>;

    template class ProjectedBSpline<float>;
    template class ProjectedBSpline<double>;
} // namespace elsa
