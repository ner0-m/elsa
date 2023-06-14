#include "JacobiPreconditioner.h"
#include "VolumeDescriptor.h"
#include "Scaling.h"

namespace elsa
{
    template <typename data_t>
    JacobiPreconditioner<data_t>::JacobiPreconditioner(const LinearOperator<data_t>& op,
                                                       bool inverse)
        : LinearOperator<data_t>{op.getDomainDescriptor(), op.getRangeDescriptor()},
          _inverseDiagonal(op.getDomainDescriptor(),
                           inverse ? 1 / diagonalFromOperator(op) : diagonalFromOperator(op))
    {
    }

    template <typename data_t>
    JacobiPreconditioner<data_t>::JacobiPreconditioner(const JacobiPreconditioner<data_t>& other)
        : LinearOperator<data_t>{*other._domainDescriptor, *other._rangeDescriptor},
          _inverseDiagonal{other._inverseDiagonal.getDomainDescriptor(),
                           other._inverseDiagonal.getScaleFactors()}
    {
    }

    template <typename data_t>
    auto JacobiPreconditioner<data_t>::cloneImpl() const -> JacobiPreconditioner<data_t>*
    {
        return new JacobiPreconditioner<data_t>(*this);
    }

    template <typename data_t>
    bool JacobiPreconditioner<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        // static_cast as type checked in base comparison
        auto otherJacobiPrecond = static_cast<const JacobiPreconditioner<data_t>*>(&other);

        if (_inverseDiagonal != otherJacobiPrecond->_inverseDiagonal)
            return false;

        return true;
    }

    template <typename data_t>
    void JacobiPreconditioner<data_t>::applyImpl(const DataContainer<data_t>& x,
                                                 DataContainer<data_t>& Ax) const
    {
        _inverseDiagonal.apply(x, Ax);
    }

    template <typename data_t>
    void JacobiPreconditioner<data_t>::applyAdjointImpl(const DataContainer<data_t>& y,
                                                        DataContainer<data_t>& Aty) const
    {
        _inverseDiagonal.applyAdjoint(y, Aty);
    }

    template <typename data_t>
    DataContainer<data_t>
        JacobiPreconditioner<data_t>::diagonalFromOperator(const LinearOperator<data_t>& op)
    {
        DataContainer<data_t> e(op.getDomainDescriptor());
        e = 0;
        DataContainer<data_t> diag(op.getDomainDescriptor());
        for (index_t i = 0; i < e.getSize(); i++) {
            e[i] = 1;
            diag[i] = op.apply(e)[i];
            e[i] = 0;
        }

        return diag;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class JacobiPreconditioner<float>;
    template class JacobiPreconditioner<double>;
} // namespace elsa
