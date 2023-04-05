#include "L2Squared.h"

#include "DataContainer.h"
#include "DataDescriptor.h"
#include "Identity.h"
#include "LinearOperator.h"
#include "TypeCasts.hpp"

namespace elsa
{
    template <typename data_t>
    L2Squared<data_t>::L2Squared(const DataDescriptor& domainDescriptor)
        : Functional<data_t>(domainDescriptor)
    {
    }

    template <typename data_t>
    L2Squared<data_t>::L2Squared(const DataContainer<data_t>& b)
        : Functional<data_t>(b.getDataDescriptor()), b_(b)
    {
    }

    template <typename data_t>
    bool L2Squared<data_t>::isDifferentiable() const
    {
        return true;
    }

    template <typename data_t>
    bool L2Squared<data_t>::hasDataVector() const
    {
        return b_.has_value();
    }

    template <typename data_t>
    const DataContainer<data_t>& L2Squared<data_t>::getDataVector() const
    {
        if (!hasDataVector()) {
            throw Error("L2Squared: No data vector present");
        }

        return *b_;
    }

    template <typename data_t>
    data_t L2Squared<data_t>::evaluateImpl(const DataContainer<data_t>& x)
    {
        if (!hasDataVector()) {
            return data_t{0.5} * x.squaredL2Norm();
        }

        return data_t{0.5} * (x - *b_).squaredL2Norm();
    }

    template <typename data_t>
    void L2Squared<data_t>::getGradientImpl(const DataContainer<data_t>& x,
                                            DataContainer<data_t>& out)
    {
        if (!hasDataVector()) {
            out = x;
        } else {
            out = x - *b_;
        }
    }

    template <typename data_t>
    LinearOperator<data_t> L2Squared<data_t>::getHessianImpl(const DataContainer<data_t>& Rx)
    {
        return leaf(Identity<data_t>(Rx.getDataDescriptor()));
    }

    template <typename data_t>
    L2Squared<data_t>* L2Squared<data_t>::cloneImpl() const
    {
        if (!hasDataVector()) {
            return new L2Squared(this->getDomainDescriptor());
        }
        return new L2Squared(*b_);
    }

    template <typename data_t>
    bool L2Squared<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (!Functional<data_t>::isEqual(other))
            return false;

        auto fn = downcast_safe<L2Squared<data_t>>(&other);
        if (!fn) {
            return false;
        }

        if (b_ && fn->b_) {
            return *b_ == *fn->b_;
        }

        return b_ == fn->b_;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class L2Squared<float>;
    template class L2Squared<double>;
    template class L2Squared<complex<double>>;
    template class L2Squared<complex<float>>;
} // namespace elsa
