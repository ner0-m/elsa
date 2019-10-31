#include "Residual.h"

#include <stdexcept>

namespace elsa
{
    template <typename data_t>
    Residual<data_t>::Residual(const DataDescriptor& domainDescriptor,
                               const DataDescriptor& rangeDescriptor)
        : _domainDescriptor{domainDescriptor.clone()}, _rangeDescriptor{rangeDescriptor.clone()}
    {
    }

    template <typename data_t>
    const DataDescriptor& Residual<data_t>::getDomainDescriptor() const
    {
        return *_domainDescriptor;
    }

    template <typename data_t>
    const DataDescriptor& Residual<data_t>::getRangeDescriptor() const
    {
        return *_rangeDescriptor;
    }

    template <typename data_t>
    DataContainer<data_t> Residual<data_t>::evaluate(const DataContainer<data_t>& x)
    {
        DataContainer<data_t> result(*_rangeDescriptor);
        evaluate(x, result);
        return result;
    }

    template <typename data_t>
    void Residual<data_t>::evaluate(const DataContainer<data_t>& x, DataContainer<data_t>& result)
    {
        if (x.getDataDescriptor() != getDomainDescriptor()
            || result.getDataDescriptor() != getRangeDescriptor())
            throw std::invalid_argument("Residual::evaluate: argument sizes do not match residual");

        _evaluate(x, result);
    }

    template <typename data_t>
    LinearOperator<data_t> Residual<data_t>::getJacobian(const DataContainer<data_t>& x)
    {
        return _getJacobian(x);
    }

    template <typename data_t>
    bool Residual<data_t>::isEqual(const Residual<data_t>& other) const
    {
        if (*_domainDescriptor != *other._domainDescriptor
            || *_rangeDescriptor != *other._rangeDescriptor)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class Residual<float>;
    template class Residual<double>;
    template class Residual<std::complex<float>>;
    template class Residual<std::complex<double>>;

} // namespace elsa
