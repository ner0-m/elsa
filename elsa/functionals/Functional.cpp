#include "Functional.h"
#include "LinearResidual.h"
#include "TypeCasts.hpp"
#include "VolumeDescriptor.h"

#include <stdexcept>

namespace elsa
{
    template <typename data_t>
    Functional<data_t>::Functional(const DataDescriptor& domainDescriptor)
        : _domainDescriptor{domainDescriptor.clone()},
          _residual{std::make_unique<LinearResidual<data_t>>(domainDescriptor)}
    {
    }

    template <typename data_t>
    Functional<data_t>::Functional(const Residual<data_t>& residual)
        : _domainDescriptor{residual.getDomainDescriptor().clone()}, _residual{residual.clone()}
    {
    }

    template <typename data_t>
    const DataDescriptor& Functional<data_t>::getDomainDescriptor() const
    {
        return *_domainDescriptor;
    }

    template <typename data_t>
    const Residual<data_t>& Functional<data_t>::getResidual() const
    {
        return *_residual;
    }

    template <typename data_t>
    data_t Functional<data_t>::evaluate(const DataContainer<data_t>& x)
    {
        if (x.getSize() != getDomainDescriptor().getNumberOfCoefficients())
            throw InvalidArgumentError(
                "Functional::evaluate: argument size does not match functional");

        // optimize for trivial LinearResiduals (no extra copy for residual result needed then)
        if (auto* linearResidual = downcast_safe<LinearResidual<data_t>>(_residual.get())) {
            if (!linearResidual->hasOperator() && !linearResidual->hasDataVector())
                return evaluateImpl(x);
        }

        // in all other cases: evaluate the residual first, then call our virtual evaluateImpl
        return evaluateImpl(_residual->evaluate(x));
    }

    template <typename data_t>
    DataContainer<data_t> Functional<data_t>::getGradient(const DataContainer<data_t>& x)
    {
        DataContainer<data_t> result(_residual->getDomainDescriptor());
        getGradient(x, result);
        return result;
    }

    template <typename data_t>
    void Functional<data_t>::getGradient(const DataContainer<data_t>& x,
                                         DataContainer<data_t>& result)
    {
        if (x.getSize() != getDomainDescriptor().getNumberOfCoefficients()
            || result.getSize() != _residual->getDomainDescriptor().getNumberOfCoefficients())
            throw InvalidArgumentError(
                "Functional::getGradient: argument sizes do not match functional");

        // optimize for trivial or simple LinearResiduals
        if (auto* linearResidual = downcast_safe<LinearResidual<data_t>>(_residual.get())) {
            // if trivial, no extra copy for residual result needed (and no chain rule)
            if (!linearResidual->hasOperator() && !linearResidual->hasDataVector()) {
                result = x;
                getGradientInPlaceImpl(result);
                return;
            }

            // if no operator, no need for chain rule
            if (!linearResidual->hasOperator()) {
                linearResidual->evaluate(x,
                                         result); // sizes of x and result will match in this case
                getGradientInPlaceImpl(result);
                return;
            }
        }

        // the general case
        auto temp = _residual->evaluate(x);
        getGradientInPlaceImpl(temp);
        _residual->getJacobian(x).applyAdjoint(temp, result); // apply the chain rule
    }

    template <typename data_t>
    LinearOperator<data_t> Functional<data_t>::getHessian(const DataContainer<data_t>& x)
    {
        // optimize for trivial and simple LinearResiduals
        if (auto* linearResidual = downcast_safe<LinearResidual<data_t>>(_residual.get())) {
            // if trivial, no extra copy for residual result needed (and no chain rule)
            if (!linearResidual->hasOperator() && !linearResidual->hasDataVector())
                return getHessianImpl(x);

            // if no operator, no need for chain rule
            if (!linearResidual->hasOperator())
                return getHessianImpl(_residual->evaluate(x));
        }

        // the general case (with chain rule)
        auto jacobian = _residual->getJacobian(x);
        auto hessian = adjoint(jacobian) * (getHessianImpl(_residual->evaluate(x))) * (jacobian);
        return hessian;
    }

    template <typename data_t>
    bool Functional<data_t>::isEqual(const Functional<data_t>& other) const
    {
        return !static_cast<bool>(*_domainDescriptor != *other._domainDescriptor
                                  || *_residual != *other._residual);
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
    void FunctionalSum<data_t>::getGradientInPlaceImpl(DataContainer<data_t>& Rx)
    {
        auto tmp = Rx;
        lhs_->getGradient(Rx, Rx);
        rhs_->getGradient(tmp, tmp);
        Rx += tmp;
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
    void FunctionalScalarMul<data_t>::getGradientInPlaceImpl(DataContainer<data_t>& Rx)
    {
        fn_->getGradient(Rx, Rx);
        Rx *= scalar_;
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
