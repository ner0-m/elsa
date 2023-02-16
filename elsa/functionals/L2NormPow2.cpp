#include "L2NormPow2.h"

#include "DataContainer.h"
#include "DataDescriptor.h"
#include "LinearOperator.h"
#include "LinearResidual.h"
#include "Identity.h"
#include "TypeCasts.hpp"

namespace elsa
{
    template <typename data_t>
    L2NormPow2<data_t>::L2NormPow2(const DataDescriptor& domainDescriptor)
        : Functional<data_t>(domainDescriptor)
    {
    }

    template <typename data_t>
    L2NormPow2<data_t>::L2NormPow2(const DataDescriptor& domainDescriptor,
                                   const DataContainer<data_t>& b)
        : Functional<data_t>(domainDescriptor), b_(b)
    {
    }

    template <typename data_t>
    L2NormPow2<data_t>::L2NormPow2(const LinearOperator<data_t>& A)
        : Functional<data_t>(A.getDomainDescriptor()), A_(A.clone())
    {
    }

    template <typename data_t>
    L2NormPow2<data_t>::L2NormPow2(const LinearOperator<data_t>& A, const DataContainer<data_t>& b)
        : Functional<data_t>(A.getDomainDescriptor()), A_(A.clone()), b_(b)
    {
    }

    template <typename data_t>
    bool L2NormPow2<data_t>::hasOperator() const
    {
        return static_cast<bool>(A_);
    }

    template <typename data_t>
    bool L2NormPow2<data_t>::hasDataVector() const
    {
        return b_.has_value();
    }

    template <typename data_t>
    const LinearOperator<data_t>& L2NormPow2<data_t>::getOperator() const
    {
        if (!hasOperator()) {
            throw Error("L2NormPow2: No operator present");
        }
        return *A_;
    }

    template <typename data_t>
    const DataContainer<data_t>& L2NormPow2<data_t>::getDataVector() const
    {
        if (!hasDataVector()) {
            throw Error("L2NormPow2: No data vector present");
        }
        return *b_;
    }

    template <typename data_t>
    data_t L2NormPow2<data_t>::evaluateImpl(const DataContainer<data_t>& Rx)
    {
        return static_cast<data_t>(0.5) * Rx.squaredL2Norm();
    }

    template <typename data_t>
    void L2NormPow2<data_t>::getGradientImpl(const DataContainer<data_t>& x,
                                             DataContainer<data_t>& out)
    {
        if (!A_ && !b_.has_value()) {
            // if no operator and no data term is present: just copy
            out = x;
        } else if (!A_) {
            // If no operator is present: no chain rule is necessary
            out = x - *b_;
        } else {
            // Apply residual Ax - b
            auto temp = A_->apply(x);
            if (b_.has_value()) {
                temp -= *b_;
            }

            // Apply chain rule
            A_->applyAdjoint(temp, out);
        }
    }

    template <typename data_t>
    LinearOperator<data_t> L2NormPow2<data_t>::getHessianImpl(const DataContainer<data_t>& Rx)
    {
        if (!A_) {
            return leaf(Identity<data_t>(Rx.getDataDescriptor()));
        } else {
            return leaf(adjoint(*A_) * (*A_));
        }
    }

    template <typename data_t>
    L2NormPow2<data_t>* L2NormPow2<data_t>::cloneImpl() const
    {
        if (A_ && b_.has_value()) {
            return new L2NormPow2(*A_, *b_);
        } else if (A_ && !b_.has_value()) {
            return new L2NormPow2(*A_);
        } else if (!A_ && b_.has_value()) {
            return new L2NormPow2(this->getDomainDescriptor(), *b_);
        } else {
            return new L2NormPow2(this->getDomainDescriptor());
        }
    }

    template <typename data_t>
    bool L2NormPow2<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (!Functional<data_t>::isEqual(other))
            return false;

        auto fn = downcast<L2NormPow2<data_t>>(&other);
        if (!fn) {
            return false;
        }

        if (b_ && fn->b_ && *b_ != *fn->b_) {
            return false;
        }

        if (A_ && fn->A_ && *A_ != *fn->A_) {
            return false;
        }

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class L2NormPow2<float>;
    template class L2NormPow2<double>;
    template class L2NormPow2<complex<float>>;
    template class L2NormPow2<complex<double>>;
} // namespace elsa
