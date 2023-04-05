#include "L2Reg.h"

#include "DataContainer.h"
#include "DataDescriptor.h"
#include "Identity.h"
#include "LinearOperator.h"
#include "TypeCasts.hpp"

namespace elsa
{
    template <typename data_t>
    L2Reg<data_t>::L2Reg(const DataDescriptor& domainDescriptor)
        : Functional<data_t>(domainDescriptor)
    {
    }

    template <typename data_t>
    L2Reg<data_t>::L2Reg(const LinearOperator<data_t>& A)
        : Functional<data_t>(A.getDomainDescriptor()), A_(A.clone())
    {
    }

    template <typename data_t>
    bool L2Reg<data_t>::isDifferentiable() const
    {
        return true;
    }

    template <typename data_t>
    bool L2Reg<data_t>::hasOperator() const
    {
        return static_cast<bool>(A_);
    }

    template <typename data_t>
    const LinearOperator<data_t>& L2Reg<data_t>::getOperator() const
    {
        if (!hasOperator()) {
            throw Error("L2Reg: No operator present");
        }

        return *A_;
    }

    template <typename data_t>
    data_t L2Reg<data_t>::evaluateImpl(const DataContainer<data_t>& x)
    {
        // If we have an operator, apply it
        if (hasOperator()) {
            auto tmp = DataContainer<data_t>(A_->getRangeDescriptor());
            A_->apply(x, tmp);

            return data_t{0.5} * tmp.squaredL2Norm();
        }

        return data_t{0.5} * x.squaredL2Norm();
    }

    template <typename data_t>
    void L2Reg<data_t>::getGradientImpl(const DataContainer<data_t>& x, DataContainer<data_t>& out)
    {
        if (hasOperator()) {
            auto temp = A_->apply(x);

            // Apply chain rule
            A_->applyAdjoint(temp, out);
        } else {
            out = x;
        }
    }

    template <typename data_t>
    LinearOperator<data_t> L2Reg<data_t>::getHessianImpl(const DataContainer<data_t>& Rx)
    {
        if (hasOperator()) {
            return leaf(adjoint(*A_) * (*A_));
        }
        return leaf(Identity<data_t>(Rx.getDataDescriptor()));
    }

    template <typename data_t>
    L2Reg<data_t>* L2Reg<data_t>::cloneImpl() const
    {
        if (hasOperator()) {
            return new L2Reg(*A_);
        }
        return new L2Reg(this->getDomainDescriptor());
    }

    template <typename data_t>
    bool L2Reg<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (!Functional<data_t>::isEqual(other))
            return false;

        auto fn = downcast_safe<L2Reg<data_t>>(&other);
        if (!fn) {
            return false;
        }

        if (A_ && fn->A_) {
            return *A_ == *fn->A_;
        }

        return A_ == fn->A_;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class L2Reg<float>;
    template class L2Reg<double>;
    template class L2Reg<complex<double>>;
    template class L2Reg<complex<float>>;
} // namespace elsa
