#include "Functional.h"
#include "DataContainer.h"
#include "TypeCasts.hpp"
#include "VolumeDescriptor.h"

#include <stdexcept>

namespace elsa
{
    template <typename data_t>
    Functional<data_t>::Functional(const DataDescriptor& domainDescriptor)
        : _domainDescriptor{domainDescriptor.clone()}
    {
    }

    template <typename data_t>
    const DataDescriptor& Functional<data_t>::getDomainDescriptor() const
    {
        return *_domainDescriptor;
    }

    template <typename data_t>
    bool Functional<data_t>::isDifferentiable() const
    {
        return false;
    }

    template <typename data_t>
    data_t Functional<data_t>::evaluate(const DataContainer<data_t>& x)
    {
        // TODO: This should compare descriptors shouldn't it?
        if (x.getSize() != getDomainDescriptor().getNumberOfCoefficients()) {
            throw InvalidArgumentError(
                "Functional::evaluate: argument size does not match functional");
        }

        return evaluateImpl(x);
    }

    template <typename data_t>
    DataContainer<data_t> Functional<data_t>::getGradient(const DataContainer<data_t>& x)
    {
        DataContainer<data_t> result(getDomainDescriptor());
        getGradient(x, result);
        return result;
    }

    template <typename data_t>
    void Functional<data_t>::getGradient(const DataContainer<data_t>& x,
                                         DataContainer<data_t>& result)
    {
        if (x.getSize() != getDomainDescriptor().getNumberOfCoefficients()) {
            throw InvalidArgumentError(
                "Functional::getGradient: argument sizes do not match functional");
        }

        getGradientImpl(x, result);
    }

    template <typename data_t>
    LinearOperator<data_t> Functional<data_t>::getHessian(const DataContainer<data_t>& x)
    {
        return getHessianImpl(x);
    }

    template <typename data_t>
    bool Functional<data_t>::isEqual(const Functional<data_t>& other) const
    {
        return !static_cast<bool>(*_domainDescriptor != *other._domainDescriptor);
    }

    // ------------------------------------------
    // FunctionalSum
    template <class data_t>
    FunctionalSum<data_t>::FunctionalSum(const Functional<data_t>& lhs,
                                         const Functional<data_t>& rhs)
        : Functional<data_t>(lhs.getDomainDescriptor()), lhs_(lhs.clone()), rhs_(rhs.clone())
    {
        if (lhs_->getDomainDescriptor() != rhs_->getDomainDescriptor()) {
            throw InvalidArgumentError("FunctionalSum: domain descriptors need to be the same");
        }
    }

    template <class data_t>
    data_t FunctionalSum<data_t>::evaluateImpl(const DataContainer<data_t>& Rx)
    {
        return lhs_->evaluate(Rx) + rhs_->evaluate(Rx);
    }

    template <class data_t>
    void FunctionalSum<data_t>::getGradientImpl(const DataContainer<data_t>& Rx,
                                                DataContainer<data_t>& out)
    {
        auto tmp = Rx;
        lhs_->getGradient(Rx, out);
        rhs_->getGradient(tmp, tmp);
        out += tmp;
    }

    template <class data_t>
    LinearOperator<data_t> FunctionalSum<data_t>::getHessianImpl(const DataContainer<data_t>& Rx)
    {
        return lhs_->getHessian(Rx) + rhs_->getHessian(Rx);
    }

    template <class data_t>
    FunctionalSum<data_t>* FunctionalSum<data_t>::cloneImpl() const
    {
        return new FunctionalSum<data_t>(*lhs_, *rhs_);
    }

    template <class data_t>
    bool FunctionalSum<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (!Functional<data_t>::isEqual(other)) {
            return false;
        }

        auto* fn = downcast<FunctionalSum<data_t>>(&other);
        return static_cast<bool>(fn) && (*lhs_) == (*fn->lhs_) && (*rhs_) == (*fn->rhs_);
    }

    // ------------------------------------------
    // FunctionalScalarMul
    template <class data_t>
    FunctionalScalarMul<data_t>::FunctionalScalarMul(const Functional<data_t>& fn,
                                                     SelfType_t<data_t> scalar)
        : Functional<data_t>(fn.getDomainDescriptor()), fn_(fn.clone()), scalar_(scalar)
    {
    }

    template <class data_t>
    data_t FunctionalScalarMul<data_t>::evaluateImpl(const DataContainer<data_t>& Rx)
    {
        return scalar_ * fn_->evaluate(Rx);
    }

    template <class data_t>
    void FunctionalScalarMul<data_t>::getGradientImpl(const DataContainer<data_t>& Rx,
                                                      DataContainer<data_t>& out)
    {
        fn_->getGradient(Rx, out);
        out *= scalar_;
    }

    template <class data_t>
    LinearOperator<data_t>
        FunctionalScalarMul<data_t>::getHessianImpl(const DataContainer<data_t>& Rx)
    {
        return scalar_ * fn_->getHessian(Rx);
    }

    template <class data_t>
    FunctionalScalarMul<data_t>* FunctionalScalarMul<data_t>::cloneImpl() const
    {
        return new FunctionalScalarMul<data_t>(*fn_, scalar_);
    }

    template <class data_t>
    bool FunctionalScalarMul<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (!Functional<data_t>::isEqual(other)) {
            return false;
        }

        auto* fn = downcast<FunctionalScalarMul<data_t>>(&other);
        return static_cast<bool>(fn) && (*fn_) == (*fn->fn_) && scalar_ == fn->scalar_;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class Functional<float>;
    template class Functional<double>;
    template class Functional<complex<float>>;
    template class Functional<complex<double>>;

    template class FunctionalSum<float>;
    template class FunctionalSum<double>;
    template class FunctionalSum<complex<float>>;
    template class FunctionalSum<complex<double>>;

    template class FunctionalScalarMul<float>;
    template class FunctionalScalarMul<double>;
    template class FunctionalScalarMul<complex<float>>;
    template class FunctionalScalarMul<complex<double>>;
} // namespace elsa
