#include "LinearResidual.h"
#include "DataContainer.h"
#include "DataDescriptor.h"
#include "Identity.h"
#include "LinearOperator.h"
#include "TypeCasts.hpp"

#include <optional>
#include <stdexcept>

namespace elsa
{
    template <typename data_t>
    LinearResidual<data_t>::LinearResidual(const DataDescriptor& descriptor)
        : domainDesc_(descriptor.clone()), rangeDesc_(descriptor.clone())
    {
    }

    template <typename data_t>
    LinearResidual<data_t>::LinearResidual(const DataContainer<data_t>& b)
        : domainDesc_(b.getDataDescriptor().clone()),
          rangeDesc_(b.getDataDescriptor().clone()),
          _dataVector(b)
    {
    }

    template <typename data_t>
    LinearResidual<data_t>::LinearResidual(const LinearOperator<data_t>& A)
        : domainDesc_(A.getDomainDescriptor().clone()),
          rangeDesc_(A.getRangeDescriptor().clone()),
          _operator(A.clone())
    {
    }

    template <typename data_t>
    LinearResidual<data_t>::LinearResidual(const LinearOperator<data_t>& A,
                                           const DataContainer<data_t>& b)
        : domainDesc_(A.getDomainDescriptor().clone()),
          rangeDesc_(A.getRangeDescriptor().clone()),
          _operator(A.clone()),
          _dataVector{b}
    {
        if (A.getRangeDescriptor() != b.getDataDescriptor())
            throw InvalidArgumentError("LinearResidual: A and b do not match");
    }

    namespace detail
    {
        template <class data_t>
        std::unique_ptr<LinearOperator<data_t>> extractOp(LinearOperator<data_t>* op)
        {
            if (op) {
                return op->clone();
            } else {
                return nullptr;
            }
        }
    } // namespace detail

    template <typename data_t>
    LinearResidual<data_t>::LinearResidual(const LinearResidual<data_t>& other)
        : domainDesc_(other.domainDesc_->clone()),
          rangeDesc_(other.rangeDesc_->clone()),
          _operator(detail::extractOp(other._operator.get())),
          _dataVector(other._dataVector)
    {
    }

    template <typename data_t>
    LinearResidual<data_t>& LinearResidual<data_t>::operator=(const LinearResidual<data_t>& other)
    {
        domainDesc_ = other.domainDesc_->clone();
        rangeDesc_ = other.rangeDesc_->clone();

        if (other.hasOperator()) {
            _operator = other._operator->clone();
        } else {
            _operator = nullptr;
        }

        if (other.hasDataVector()) {
            _dataVector = other.getDataVector();
        } else {
            _dataVector = std::nullopt;
        }
        return *this;
    }

    template <typename data_t>
    LinearResidual<data_t>::LinearResidual(LinearResidual<data_t>&& other) noexcept
        : domainDesc_(std::move(other.domainDesc_)),
          rangeDesc_(std::move(other.rangeDesc_)),
          _operator(std::move(other._operator)),
          _dataVector(std::move(other._dataVector))
    {
    }

    template <typename data_t>
    LinearResidual<data_t>&
        LinearResidual<data_t>::operator=(LinearResidual<data_t>&& other) noexcept
    {
        domainDesc_ = std::move(other.domainDesc_);
        rangeDesc_ = std::move(other.rangeDesc_);
        _operator = std::move(other._operator);
        _dataVector = std::move(other._dataVector);

        return *this;
    }

    template <typename data_t>
    const DataDescriptor& LinearResidual<data_t>::getDomainDescriptor() const
    {
        return *domainDesc_;
    }

    template <typename data_t>
    const DataDescriptor& LinearResidual<data_t>::getRangeDescriptor() const
    {
        return *rangeDesc_;
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
    DataContainer<data_t> LinearResidual<data_t>::evaluate(const DataContainer<data_t>& x) const
    {
        DataContainer<data_t> out(this->getRangeDescriptor());
        evaluate(x, out);
        return out;
    }

    template <typename data_t>
    void LinearResidual<data_t>::evaluate(const DataContainer<data_t>& x,
                                          DataContainer<data_t>& result) const
    {
        if (hasOperator())
            _operator->apply(x, result);
        else
            result = x;

        if (hasDataVector()) {
            result -= *_dataVector;
        }
    }

    template <typename data_t>
    LinearOperator<data_t>
        LinearResidual<data_t>::getJacobian([[maybe_unused]] const DataContainer<data_t>& x)
    {
        if (hasOperator())
            return leaf(*_operator);
        else
            return leaf(Identity<data_t>(this->getRangeDescriptor()));
    }

    // ------------------------------------------
    // explicit template instantiation
    template class LinearResidual<float>;
    template class LinearResidual<double>;
    template class LinearResidual<complex<float>>;
    template class LinearResidual<complex<double>>;

} // namespace elsa
