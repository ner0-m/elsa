#include "Functional.h"
#include "LinearResidual.h"

#include <stdexcept>

namespace elsa
{
    template <typename data_t>
    Functional<data_t>::Functional(const DataDescriptor& domainDescriptor)
        : _domainDescriptor{domainDescriptor.clone()},
          _residual{std::make_unique<LinearResidual<data_t>>(domainDescriptor)}
    {}

    template <typename data_t>
    Functional<data_t>::Functional(const Residual<data_t>& residual)
        : _domainDescriptor{residual.getDomainDescriptor().clone()},
          _residual{residual.clone()}
    {}

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
        if (x.getDataDescriptor() != getDomainDescriptor())
            throw std::invalid_argument("Functional::evaluate: argument size does not match functional");

        // optimize for trivial LinearResiduals (no extra copy for residual result needed then)
        if (auto* linearResidual = dynamic_cast<LinearResidual<data_t>*>(_residual.get())) {
            if (!linearResidual->hasOperator() && !linearResidual->hasDataVector())
                return _evaluate(x);
        }

        // in all other cases: evaluate the residual first, then call our virtual _evaluate
        return _evaluate(_residual->evaluate(x));
    }

    template <typename data_t>
    DataContainer<data_t> Functional<data_t>::getGradient(const DataContainer<data_t>& x)
    {
        DataContainer<data_t> result(_residual->getRangeDescriptor());
        getGradient(x, result);
        return result;
    }

    template <typename data_t>
    void Functional<data_t>::getGradient(const DataContainer<data_t>& x, DataContainer<data_t>& result)
    {
        if (x.getDataDescriptor() != getDomainDescriptor() ||
            result.getDataDescriptor() != _residual->getRangeDescriptor())
            throw std::invalid_argument("Functional::getGradient: argument sizes do not match functional");

        // optimize for trivial or simple LinearResiduals
        if (auto* linearResidual = dynamic_cast<LinearResidual<data_t>*>(_residual.get())) {
            // if trivial, no extra copy for residual result needed (and no chain rule)
            if (!linearResidual->hasOperator() && !linearResidual->hasDataVector()) {
                result = x;
                _getGradientInPlace(result);
                return;
            }

            // if no operator, no need for chain rule
            if (!linearResidual->hasOperator()) {
                linearResidual->evaluate(x, result); // sizes of x and result will match in this case
                _getGradientInPlace(result);
                return;
            }
        }

        // the general case
        auto temp = _residual->evaluate(x);
        _getGradientInPlace(temp);
        _residual->getJacobian(x).applyAdjoint(temp, result); // apply the chain rule
    }

    template <typename data_t>
    LinearOperator<data_t> Functional<data_t>::getHessian(const DataContainer<data_t>& x)
    {
        // optimize for trivial and simple LinearResiduals
        if (auto* linearResidual = dynamic_cast<LinearResidual<data_t>*>(_residual.get())) {
            // if trivial, no extra copy for residual result needed (and no chain rule)
            if (!linearResidual->hasOperator() && !linearResidual->hasDataVector())
                return _getHessian(x);

            // if no operator, no need for chain rule
            if (!linearResidual->hasOperator())
                return _getHessian(_residual->evaluate(x));
        }

        // the general case (with chain rule)
        auto jacobian = _residual->getJacobian(x);
        auto hessian = adjoint(jacobian) * (_getHessian(_residual->evaluate(x))) * (jacobian);
        return hessian;
    }


    template <typename data_t>
    bool Functional<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (*_domainDescriptor != *other._domainDescriptor ||
            *_residual != *other._residual)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class Functional<float>;
    template class Functional<double>;
    template class Functional<std::complex<float>>;
    template class Functional<std::complex<double>>;

} // namespace elsa
