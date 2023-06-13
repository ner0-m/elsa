#include "WeightedLeastSquares.h"

#include "DataContainer.h"
#include "DataDescriptor.h"
#include "LinearOperator.h"
#include "Scaling.h"
#include "TypeCasts.hpp"

namespace elsa
{
    template <typename data_t>
    WeightedLeastSquares<data_t>::WeightedLeastSquares(const LinearOperator<data_t>& A,
                                                       const DataContainer<data_t>& b,
                                                       const DataContainer<data_t>& weights)
        : Functional<data_t>(A.getDomainDescriptor()),
          A_(A.clone()),
          b_(b),
          W_(A.getDomainDescriptor(), weights)
    {
    }

    template <typename data_t>
    bool WeightedLeastSquares<data_t>::isDifferentiable() const
    {
        return true;
    }

    template <typename data_t>
    const LinearOperator<data_t>& WeightedLeastSquares<data_t>::getOperator() const
    {
        return *A_;
    }

    template <typename data_t>
    const DataContainer<data_t>& WeightedLeastSquares<data_t>::getDataVector() const
    {
        return b_;
    }

    template <typename data_t>
    data_t WeightedLeastSquares<data_t>::evaluateImpl(const DataContainer<data_t>& x)
    {
        // Evaluate A(x) - b
        auto temp = A_->apply(x);
        temp -= b_;

        // evaluate weighted l2 norm
        W_.apply(x, temp);
        return static_cast<data_t>(0.5) * x.dot(temp);
    }

    template <typename data_t>
    void WeightedLeastSquares<data_t>::getGradientImpl(const DataContainer<data_t>& x,
                                                       DataContainer<data_t>& out)
    {
        // Evaluate A(x) - b
        auto temp = A_->apply(x);
        temp -= b_;

        W_.apply(x, temp);

        // Apply chain rule
        A_->applyAdjoint(temp, out);
    }

    template <typename data_t>
    LinearOperator<data_t>
        WeightedLeastSquares<data_t>::getHessianImpl(const DataContainer<data_t>& x)
    {
        return leaf(adjoint(*A_) * W_ * (*A_));
    }

    template <typename data_t>
    WeightedLeastSquares<data_t>* WeightedLeastSquares<data_t>::cloneImpl() const
    {
        return new WeightedLeastSquares(*A_, b_, W_.getScaleFactors());
    }

    template <typename data_t>
    bool WeightedLeastSquares<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (!Functional<data_t>::isEqual(other))
            return false;

        auto fn = downcast_safe<WeightedLeastSquares<data_t>>(&other);
        return fn && *A_ == *fn->A_ && b_ == fn->b_ && W_ == fn->W_;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class WeightedLeastSquares<float>;
    template class WeightedLeastSquares<double>;
} // namespace elsa
