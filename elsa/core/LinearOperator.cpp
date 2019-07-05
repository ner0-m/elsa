#include "LinearOperator.h"

#include <stdexcept>
#include <typeinfo>

namespace elsa
{
    template <typename data_t>
    LinearOperator<data_t>::LinearOperator(const DataDescriptor& domainDescriptor, const DataDescriptor& rangeDescriptor)
        : _domainDescriptor{domainDescriptor.clone()}, _rangeDescriptor{rangeDescriptor.clone()}
    {}

    template <typename data_t>
    LinearOperator<data_t>::LinearOperator(const LinearOperator<data_t>& other)
        : _domainDescriptor{other._domainDescriptor->clone()}, _rangeDescriptor{other._rangeDescriptor->clone()},
          _isLeaf{other._isLeaf}, _isAdjoint{other._isAdjoint}, _isComposite{other._isComposite}, _mode{other._mode}
    {
        if (_isLeaf)
            _lhs = other._lhs->clone();

        if (_isComposite) {
            _lhs = other._lhs->clone();
            _rhs = other._rhs->clone();
        }
    }

    template <typename data_t>
    LinearOperator<data_t>& LinearOperator<data_t>::operator=(const LinearOperator<data_t>& other)
    {
        if (*this != other) {
            _domainDescriptor = other._domainDescriptor->clone();
            _rangeDescriptor = other._rangeDescriptor->clone();
            _isLeaf = other._isLeaf;
            _isAdjoint = other._isAdjoint;
            _isComposite = other._isComposite;
            _mode = other._mode;

            if (_isLeaf)
                _lhs = other._lhs->clone();

            if (_isComposite) {
                _lhs = other._lhs->clone();
                _rhs = other._rhs->clone();
            }
        }

        return *this;
    }


    template <typename data_t>
    const DataDescriptor& LinearOperator<data_t>::getDomainDescriptor() const
    {
        return *_domainDescriptor;
    }

    template <typename data_t>
    const DataDescriptor& LinearOperator<data_t>::getRangeDescriptor() const
    {
        return *_rangeDescriptor;
    }


    template <typename data_t>
    DataContainer<data_t> LinearOperator<data_t>::apply(const DataContainer<data_t>& x)
    {
        DataContainer<data_t> result(*_rangeDescriptor);
        apply(x, result);
        return result;
    }
    
    template <typename data_t>
    void LinearOperator<data_t>::apply(const DataContainer<data_t>& x, DataContainer<data_t>& Ax)
    {
        _apply(x, Ax);
    }

    template <typename data_t>
    void LinearOperator<data_t>::_apply(const DataContainer<data_t>& x, DataContainer<data_t>& Ax)
    {
        if (_isLeaf) {
            if (_isAdjoint) {
                // sanity check the arguments for the intended evaluation tree leaf operation
                if (_lhs->getRangeDescriptor().getNumberOfCoefficients() != x.getSize() ||
                    _lhs->getDomainDescriptor().getNumberOfCoefficients() != Ax.getSize())
                    throw std::invalid_argument("LinearOperator::apply: incorrect input/output sizes for adjoint leaf");

                _lhs->applyAdjoint(x, Ax);
                return;
            }
            else {
                // sanity check the arguments for the intended evaluation tree leaf operation
                if (_lhs->getDomainDescriptor().getNumberOfCoefficients() != x.getSize() ||
                    _lhs->getRangeDescriptor().getNumberOfCoefficients() != Ax.getSize())
                    throw std::invalid_argument("LinearOperator::apply: incorrect input/output sizes for leaf");

                _lhs->apply(x, Ax);
                return;
            }
        }

        if (_isComposite) {
            if (_mode == compositeMode::add) {
                // sanity check the arguments for the intended evaluation tree leaf operation
                if (_rhs->getDomainDescriptor().getNumberOfCoefficients() != x.getSize() ||
                    _rhs->getRangeDescriptor().getNumberOfCoefficients() != Ax.getSize() ||
                    _lhs->getDomainDescriptor().getNumberOfCoefficients() != x.getSize() ||
                    _lhs->getRangeDescriptor().getNumberOfCoefficients() != Ax.getSize())
                    throw std::invalid_argument("LinearOperator::apply: incorrect input/output sizes for add leaf");

                _rhs->apply(x, Ax);
                Ax += _lhs->apply(x);
                return;
            }

            if (_mode == compositeMode::mult) {
                // sanity check the arguments for the intended evaluation tree leaf operation
                if (_rhs->getDomainDescriptor().getNumberOfCoefficients() != x.getSize() ||
                    _lhs->getRangeDescriptor().getNumberOfCoefficients() != Ax.getSize())
                    throw std::invalid_argument("LinearOperator::apply: incorrect input/output sizes for mult leaf");

                DataContainer<data_t> temp(_rhs->getRangeDescriptor());
                _rhs->apply(x, temp);
                _lhs->apply(temp, Ax);
                return;
            }

        }

        throw std::logic_error("LinearOperator: apply called on ill-formed object");
    }


    template <typename data_t>
    DataContainer<data_t> LinearOperator<data_t>::applyAdjoint(const DataContainer<data_t>& y)
    {
        DataContainer<data_t> result(*_domainDescriptor);
        applyAdjoint(y, result);
        return result;
    }
    
    template <typename data_t>
    void LinearOperator<data_t>::applyAdjoint(const DataContainer<data_t>& y, DataContainer<data_t>& Aty)
    {
        _applyAdjoint(y, Aty);
    }

    template <typename data_t>
    void LinearOperator<data_t>::_applyAdjoint(const DataContainer<data_t>& y, DataContainer<data_t>& Aty)
    {
        if (_isLeaf) {
            if (_isAdjoint) {
                // sanity check the arguments for the intended evaluation tree leaf operation
                if (_lhs->getDomainDescriptor().getNumberOfCoefficients() != y.getSize() ||
                    _lhs->getRangeDescriptor().getNumberOfCoefficients() != Aty.getSize())
                    throw std::invalid_argument(
                            "LinearOperator::applyAdjoint: incorrect input/output sizes for adjoint leaf");

                _lhs->apply(y, Aty);
                return;
            }
            else {
                // sanity check the arguments for the intended evaluation tree leaf operation
                if (_lhs->getRangeDescriptor().getNumberOfCoefficients() != y.getSize() ||
                    _lhs->getDomainDescriptor().getNumberOfCoefficients() != Aty.getSize())
                    throw std::invalid_argument("LinearOperator::applyAdjoint: incorrect input/output sizes for leaf");

                _lhs->applyAdjoint(y, Aty);
                return;
            }
        }

        if (_isComposite) {
            if (_mode == compositeMode::add) {
                // sanity check the arguments for the intended evaluation tree leaf operation
                if (_rhs->getRangeDescriptor().getNumberOfCoefficients() != y.getSize() ||
                    _rhs->getDomainDescriptor().getNumberOfCoefficients() != Aty.getSize() ||
                    _lhs->getRangeDescriptor().getNumberOfCoefficients() != y.getSize() ||
                    _lhs->getDomainDescriptor().getNumberOfCoefficients() != Aty.getSize())
                    throw std::invalid_argument("LinearOperator::applyAdjoint: incorrect input/output sizes for add leaf");

                _rhs->applyAdjoint(y, Aty);
                Aty += _lhs->applyAdjoint(y);
                return;
            }

            if (_mode == compositeMode::mult) {
                // sanity check the arguments for the intended evaluation tree leaf operation
                if (_lhs->getRangeDescriptor().getNumberOfCoefficients() != y.getSize() ||
                    _rhs->getDomainDescriptor().getNumberOfCoefficients() != Aty.getSize())
                    throw std::invalid_argument(
                            "LinearOperator::applyAdjoint: incorrect input/output sizes for mult leaf");

                DataContainer<data_t> temp(_lhs->getDomainDescriptor());
                _lhs->applyAdjoint(y, temp);
                _rhs->applyAdjoint(temp, Aty);
                return;
            }
        }

        throw std::logic_error("LinearOperator: applyAdjoint called on ill-formed object");
    }

    template <typename data_t>
    LinearOperator<data_t>* LinearOperator<data_t>::cloneImpl() const
    {
        if (_isLeaf)
            return new LinearOperator<data_t>(*_lhs, _isAdjoint);

        if (_isComposite)
            return new LinearOperator<data_t>(*_lhs, *_rhs, _mode);

        return new LinearOperator<data_t>(*_domainDescriptor, *_rangeDescriptor);
    }

    template <typename data_t>
    bool LinearOperator<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {
        if (*_domainDescriptor != *other._domainDescriptor ||
            *_rangeDescriptor != *other._rangeDescriptor)
            return false;

        if (_isLeaf)
            return other._isLeaf && (_isAdjoint == other._isAdjoint) && (*_lhs == *other._lhs);

        if (_isComposite)
            return other._isComposite && _mode == other._mode
                && (*_lhs == *other._lhs) && (*_rhs == *other._rhs);

        return true;
    }

    template <typename data_t>
    LinearOperator<data_t>::LinearOperator(const LinearOperator<data_t>& op, bool isAdjoint)
        : _domainDescriptor{(isAdjoint) ? op.getRangeDescriptor().clone() : op.getDomainDescriptor().clone()},
          _rangeDescriptor{(isAdjoint) ? op.getDomainDescriptor().clone() : op.getRangeDescriptor().clone()},
          _lhs{op.clone()}, _isLeaf{true}, _isAdjoint{isAdjoint}
    {
    }

    template <typename data_t>
    LinearOperator<data_t>::LinearOperator(const LinearOperator<data_t>& lhs,
            const LinearOperator<data_t>& rhs, compositeMode mode)
            : _domainDescriptor{rhs.getDomainDescriptor().clone()}, _rangeDescriptor{lhs.getRangeDescriptor().clone()},
              _lhs{lhs.clone()}, _rhs{rhs.clone()}, _isComposite{true}, _mode{mode}
    {
        // sanity check the descriptors
        switch (_mode) {
            case compositeMode::add:
                // for addition, both domains and ranges should match
                if (_lhs->getDomainDescriptor() != _rhs->getDomainDescriptor() ||
                    _lhs->getRangeDescriptor() != _rhs->getRangeDescriptor())
                    throw std::invalid_argument("LinearOperator: composite add domain/range mismatch");
                break;

            case compositeMode::mult:
                // for multiplication, domain of _lhs should match range of _rhs
                if (_lhs->getDomainDescriptor() != _rhs->getRangeDescriptor())
                    throw std::invalid_argument("LinearOperator: composite mult domain/range mismatch");
                break;

            default:
                throw std::logic_error("LinearOperator: unknown composition mode");
        }
    }


    // ------------------------------------------
    // explicit template instantiation
    template class LinearOperator<float>;
    template class LinearOperator<std::complex<float>>;
    template class LinearOperator<double>;
    template class LinearOperator<std::complex<double>>;

} // namespace elsa
