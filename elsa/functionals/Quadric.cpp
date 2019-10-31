#include "Quadric.h"
#include "Identity.h"

#include <stdexcept>

namespace elsa
{
    template <typename data_t>
    Quadric<data_t>::Quadric(const LinearOperator<data_t>& A, const DataContainer<data_t>& b)
        : Functional<data_t>(A.getDomainDescriptor()), _linearResidual{A, b}
    {
    }

    template <typename data_t>
    Quadric<data_t>::Quadric(const LinearOperator<data_t>& A)
        : Functional<data_t>(A.getDomainDescriptor()), _linearResidual{A}
    {
    }

    template <typename data_t>
    Quadric<data_t>::Quadric(const DataContainer<data_t>& b)
        : Functional<data_t>(b.getDataDescriptor()), _linearResidual{b}
    {
    }

    template <typename data_t>
    Quadric<data_t>::Quadric(const DataDescriptor& domainDescriptor)
        : Functional<data_t>(domainDescriptor), _linearResidual{domainDescriptor}
    {
    }

    template <typename data_t>
    const LinearResidual<data_t>& Quadric<data_t>::getGradientExpression() const
    {
        return _linearResidual;
    }

    template <typename data_t>
    data_t Quadric<data_t>::_evaluate(const DataContainer<data_t>& Rx)
    {
        data_t xtAx;

        if (_linearResidual.hasOperator()) {
            auto temp = _linearResidual.getOperator().apply(Rx);
            xtAx = Rx.dot(temp);
        } else {
            xtAx = Rx.squaredL2Norm();
        }

        if (_linearResidual.hasDataVector()) {
            return static_cast<data_t>(0.5) * xtAx - Rx.dot(_linearResidual.getDataVector());
        } else {
            return static_cast<data_t>(0.5) * xtAx;
        }
    }

    template <typename data_t>
    void Quadric<data_t>::_getGradientInPlace(DataContainer<data_t>& Rx)
    {
        Rx = _linearResidual.evaluate(Rx);
    }

    template <typename data_t>
    LinearOperator<data_t> Quadric<data_t>::_getHessian(const DataContainer<data_t>& Rx)
    {
        if (_linearResidual.hasOperator())
            return leaf(_linearResidual.getOperator());
        else
            return leaf(Identity<data_t>(*_domainDescriptor));
    }

    template <typename data_t>
    Quadric<data_t>* Quadric<data_t>::cloneImpl() const
    {
        if (_linearResidual.hasOperator() && _linearResidual.hasDataVector())
            return new Quadric<data_t>(_linearResidual.getOperator(),
                                       _linearResidual.getDataVector());
        else if (_linearResidual.hasOperator() && !_linearResidual.hasDataVector())
            return new Quadric<data_t>(_linearResidual.getOperator());
        else if (!_linearResidual.hasOperator() && _linearResidual.hasDataVector())
            return new Quadric<data_t>(_linearResidual.getDataVector());
        else
            return new Quadric<data_t>(*_domainDescriptor);
    }

    template <typename data_t>
    bool Quadric<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (!Functional<data_t>::isEqual(other))
            return false;

        auto otherQuadric = dynamic_cast<const Quadric*>(&other);
        if (!otherQuadric)
            return false;

        if (_linearResidual != otherQuadric->_linearResidual)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class Quadric<float>;
    template class Quadric<double>;
    template class Quadric<std::complex<float>>;
    template class Quadric<std::complex<double>>;

} // namespace elsa
