#include "LinearResidual.h"
#include "Identity.h"
#include "TypeCasts.hpp"

namespace elsa
{
    template <typename data_t>
    LinearResidual<data_t>::LinearResidual(const DataDescriptor& descriptor)
        : Residual<data_t>(descriptor, descriptor)
    {
    }

    template <typename data_t>
    LinearResidual<data_t>::LinearResidual(const DataContainer<data_t>& b)
        : Residual<data_t>(b.getDataDescriptor(), b.getDataDescriptor()), _dataVector{b}
    {
    }

    template <typename data_t>
    LinearResidual<data_t>::LinearResidual(const LinearOperator<data_t>& A)
        : Residual<data_t>(A.getDomainDescriptor(), A.getRangeDescriptor()), _operator{A.clone()}
    {
    }

    template <typename data_t>
    LinearResidual<data_t>::LinearResidual(const LinearOperator<data_t>& A,
                                           const DataContainer<data_t>& b)
        : Residual<data_t>(A.getDomainDescriptor(), A.getRangeDescriptor()),
          _operator{A.clone()},
          _dataVector{b}
    {
        if (A.getRangeDescriptor().getNumberOfCoefficients() != b.getSize())
            throw InvalidArgumentError("LinearResidual: A and b do not match");
    }

    template <typename data_t>
    bool LinearResidual<data_t>::hasOperator() const
    {
        return static_cast<bool>(_operator);
    }

    template <typename data_t>
    bool LinearResidual<data_t>::hasDataVector() const
    {
        return _dataVector.has_value();
    }

    template <typename data_t>
    const LinearOperator<data_t>& LinearResidual<data_t>::getOperator() const
    {
        if (!_operator)
            throw Error("LinearResidual::getOperator: operator not present");

        return *_operator;
    }

    template <typename data_t>
    const DataContainer<data_t>& LinearResidual<data_t>::getDataVector() const
    {
        if (!_dataVector)
            throw Error("LinearResidual::getDataVector: data vector not present");

        return *_dataVector;
    }

    template <typename data_t>
    LinearResidual<data_t>* LinearResidual<data_t>::cloneImpl() const
    {
        if (hasOperator() && hasDataVector())
            return new LinearResidual<data_t>(getOperator(), getDataVector());

        if (hasOperator())
            return new LinearResidual<data_t>(getOperator());

        if (hasDataVector())
            return new LinearResidual<data_t>(getDataVector());

        return new LinearResidual<data_t>(this->getDomainDescriptor());
    }

    template <typename data_t>
    bool LinearResidual<data_t>::isEqual(const Residual<data_t>& other) const
    {
        if (!Residual<data_t>::isEqual(other))
            return false;

        auto otherLinearResidual = downcast_safe<LinearResidual>(&other);
        if (!otherLinearResidual)
            return false;

        if (hasOperator() != otherLinearResidual->hasOperator()
            || hasDataVector() != otherLinearResidual->hasDataVector())
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
                                              DataContainer<data_t>& result) const
    {
        if (hasOperator())
            _operator->apply(x, result);
        else
            result = x;

        if (hasDataVector())
            result -= *_dataVector;
    }

    template <typename data_t>
    std::unique_ptr<LinearOperator<data_t>>
        LinearResidual<data_t>::getJacobianImpl([[maybe_unused]] const DataContainer<data_t>& x)
    {
        if (hasOperator())
            return _operator->clone();
        else
            return std::make_unique<Identity<data_t>>(this->getRangeDescriptor());
    }

    // ------------------------------------------
    // explicit template instantiation
    template class LinearResidual<float>;
    template class LinearResidual<double>;
    template class LinearResidual<complex<float>>;
    template class LinearResidual<complex<double>>;

} // namespace elsa
