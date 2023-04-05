#include "LeastSquares.h"

#include "DataContainer.h"
#include "DataDescriptor.h"
#include "LinearOperator.h"
#include "TypeCasts.hpp"

namespace elsa
{
    template <typename data_t>
    LeastSquares<data_t>::LeastSquares(const LinearOperator<data_t>& A,
                                       const DataContainer<data_t>& b)
        : Functional<data_t>(A.getDomainDescriptor()), A_(A.clone()), b_(b)
    {
    }

    template <typename data_t>
    bool LeastSquares<data_t>::isDifferentiable() const
    {
        return true;
    }

    template <typename data_t>
    const LinearOperator<data_t>& LeastSquares<data_t>::getOperator() const
    {
        return *A_;
    }

    template <typename data_t>
    const DataContainer<data_t>& LeastSquares<data_t>::getDataVector() const
    {
        return b_;
    }

    template <typename data_t>
    data_t LeastSquares<data_t>::evaluateImpl(const DataContainer<data_t>& x)
    {
        auto Ax = A_->apply(x);
        Ax -= b_;

        return static_cast<data_t>(0.5) * Ax.squaredL2Norm();
    }

    template <typename data_t>
    void LeastSquares<data_t>::getGradientImpl(const DataContainer<data_t>& x,
                                               DataContainer<data_t>& out)
    {
        auto temp = A_->apply(x);
        temp -= b_;

        // Apply chain rule
        A_->applyAdjoint(temp, out);
    }

    template <typename data_t>
    LinearOperator<data_t> LeastSquares<data_t>::getHessianImpl(const DataContainer<data_t>& Rx)
    {
        return leaf(adjoint(*A_) * (*A_));
    }

    template <typename data_t>
    LeastSquares<data_t>* LeastSquares<data_t>::cloneImpl() const
    {
        return new LeastSquares(*A_, b_);
    }

    template <typename data_t>
    bool LeastSquares<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (!Functional<data_t>::isEqual(other))
            return false;

        auto fn = downcast_safe<LeastSquares<data_t>>(&other);
        return fn && *A_ == *fn->A_ && b_ == fn->b_;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class LeastSquares<float>;
    template class LeastSquares<double>;
} // namespace elsa
