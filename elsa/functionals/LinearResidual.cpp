#include "LinearResidual.h"
#include "Identity.h"

#include <stdexcept>

namespace elsa
{
    template <typename data_t>
    LinearResidual<data_t>::LinearResidual(const DataDescriptor& descriptor)
        : Residual<data_t>(descriptor, descriptor), _hasOperator{false}, _hasDataVector{false}
    {
    }

    template <typename data_t>
    LinearResidual<data_t>::LinearResidual(const DataContainer<data_t>& b)
        : Residual<data_t>(b.getDataDescriptor(), b.getDataDescriptor()),
          _hasOperator{false},
          _hasDataVector{true},
          _dataVector{std::make_unique<DataContainer<data_t>>(b)}
    {
    }

    template <typename data_t>
    LinearResidual<data_t>::LinearResidual(const LinearOperator<data_t>& A)
        : Residual<data_t>(A.getDomainDescriptor(), A.getRangeDescriptor()),
          _hasOperator{true},
          _hasDataVector{false},
          _operator{A.clone()}
    {
    }

    template <typename data_t>
    LinearResidual<data_t>::LinearResidual(const LinearOperator<data_t>& A,
                                           const DataContainer<data_t>& b)
        : Residual<data_t>(A.getDomainDescriptor(), A.getRangeDescriptor()),
          _hasOperator{true},
          _hasDataVector{true},
          _operator{A.clone()},
          _dataVector{std::make_unique<DataContainer<data_t>>(b)}
    {
        if (A.getRangeDescriptor() != b.getDataDescriptor())
            throw std::invalid_argument("LinearResidual: A and b do not match");
    }

    template <typename data_t>
    bool LinearResidual<data_t>::hasOperator() const
    {
        return _hasOperator;
    }

    template <typename data_t>
    bool LinearResidual<data_t>::hasDataVector() const
    {
        return _hasDataVector;
    }

    template <typename data_t>
    const LinearOperator<data_t>& LinearResidual<data_t>::getOperator() const
    {
        if (!_operator)
            throw std::runtime_error("LinearResidual::getOperator: operator not present");

        return *_operator;
    }

    template <typename data_t>
    const DataContainer<data_t>& LinearResidual<data_t>::getDataVector() const
    {
        if (!_dataVector)
            throw std::runtime_error("LinearResidual::getDataVector: data vector not present");

        return *_dataVector;
    }

    template <typename data_t>
    LinearResidual<data_t>* LinearResidual<data_t>::cloneImpl() const
    {
        if (_hasOperator && _hasDataVector)
            return new LinearResidual<data_t>(getOperator(), getDataVector());

        if (_hasOperator)
            return new LinearResidual<data_t>(getOperator());

        if (_hasDataVector)
            return new LinearResidual<data_t>(getDataVector());

        return new LinearResidual<data_t>(this->getDomainDescriptor());
    }

    template <typename data_t>
    bool LinearResidual<data_t>::isEqual(const Residual<data_t>& other) const
    {
        if (!Residual<data_t>::isEqual(other))
            return false;

        auto otherLinearResidual = dynamic_cast<const LinearResidual*>(&other);
        if (!otherLinearResidual)
            return false;

        if (_hasOperator != otherLinearResidual->_hasOperator
            || _hasDataVector != otherLinearResidual->_hasDataVector)
            return false;

        if ((_operator && !otherLinearResidual->_operator)
            || (!_operator && otherLinearResidual->_operator)
            || (_dataVector && !otherLinearResidual->_dataVector)
            || (!_dataVector && otherLinearResidual->_dataVector))
            return false;

        if (_operator && otherLinearResidual->_operator
            && *_operator != *otherLinearResidual->_operator)
            return false;

        if (_dataVector && otherLinearResidual->_dataVector
            && *_dataVector != *otherLinearResidual->_dataVector)
            return false;

        return true;
    }

    template <typename data_t>
    void LinearResidual<data_t>::evaluateImpl(const DataContainer<data_t>& x,
                                              DataContainer<data_t>& result)
    {
        if (_hasOperator)
            _operator->apply(x, result);
        else
            result = x;

        if (_hasDataVector)
            result -= *_dataVector;
    }

    template <typename data_t>
    LinearOperator<data_t>
        LinearResidual<data_t>::getJacobianImpl([[maybe_unused]] const DataContainer<data_t>& x)
    {
        if (_hasOperator)
            return leaf(*_operator);
        else
            return leaf(Identity<data_t>(this->getRangeDescriptor()));
    }

    // ------------------------------------------
    // explicit template instantiation
    template class LinearResidual<float>;
    template class LinearResidual<double>;
    template class LinearResidual<std::complex<float>>;
    template class LinearResidual<std::complex<double>>;

} // namespace elsa
