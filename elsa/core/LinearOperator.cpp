#include "LinearOperator.h"

#include <stdexcept>
#include <typeinfo>
#include <iostream>

#include "DescriptorUtils.h"

namespace elsa
{
    template <typename data_t>
    LinearOperator<data_t>::LinearOperator(const DataDescriptor& domainDescriptor,
                                           const DataDescriptor& rangeDescriptor)
        : _domainDescriptor{domainDescriptor.clone()}, _rangeDescriptor{rangeDescriptor.clone()}
    {
    }

    template <typename data_t>
    LinearOperator<data_t>::LinearOperator(const LinearOperator<data_t>& other)
        : Cloneable<LinearOperator<data_t>>(),
          _domainDescriptor{other._domainDescriptor->clone()},
          _rangeDescriptor{other._rangeDescriptor->clone()}
    {
    }

    template <typename data_t>
    LinearOperator<data_t>& LinearOperator<data_t>::operator=(const LinearOperator<data_t>& other)
    {
        if (*this != other) {
            _domainDescriptor = other._domainDescriptor->clone();
            _rangeDescriptor = other._rangeDescriptor->clone();
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
    DataContainer<data_t> LinearOperator<data_t>::apply(const DataContainer<data_t>& x) const
    {
        DataContainer<data_t> result(*_rangeDescriptor, x.getDataHandlerType());
        apply(x, result);
        return result;
    }

    template <typename data_t>
    void LinearOperator<data_t>::apply(const DataContainer<data_t>& x,
                                       DataContainer<data_t>& Ax) const
    {
        if (getDomainDescriptor().getNumberOfCoefficients() != x.getSize()) {
            throw Error(
                "LinearOperator::apply: expected differently sized input x (expected {}, is {})",
                getDomainDescriptor().getNumberOfCoefficients(), x.getSize());
        }

        if (getRangeDescriptor().getNumberOfCoefficients() != Ax.getSize()) {
            throw Error(
                "LinearOperator::apply: expected differently sized input Ax (expected {}, is {})",
                getRangeDescriptor().getNumberOfCoefficients(), Ax.getSize());
        }

        applyImpl(x, Ax);
    }

    template <typename data_t>
    DataContainer<data_t> LinearOperator<data_t>::applyAdjoint(const DataContainer<data_t>& y) const
    {
        DataContainer<data_t> result(*_domainDescriptor, y.getDataHandlerType());
        applyAdjoint(y, result);
        return result;
    }

    template <typename data_t>
    void LinearOperator<data_t>::applyAdjoint(const DataContainer<data_t>& y,
                                              DataContainer<data_t>& Aty) const
    {
        // if (getRangeDescriptor().getNumberOfCoefficients() != y.getSize()
        //     || getDomainDescriptor().getNumberOfCoefficients() != Aty.getSize())
        //     throw InvalidArgumentError(
        //         "LinearOperator::applyAdjoint: incorrect input/output sizes for leaf");

        applyAdjointImpl(y, Aty);
    }

    template <typename data_t>
    bool LinearOperator<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {
        if (typeid(other) != typeid(*this))
            return false;

        if (*_domainDescriptor != *other._domainDescriptor
            || *_rangeDescriptor != *other._rangeDescriptor)
            return false;

        return true;
    }

    // ------------------------------------------
    // Implementation CompositeAddLinearOperator
    template <typename data_t>
    AdjointLinearOperator<data_t>::AdjointLinearOperator(const LinearOperator<data_t>& op)
        : LinearOperator<data_t>(op.getRangeDescriptor(), op.getDomainDescriptor()), op_(op.clone())
    {
    }

    template <typename data_t>
    void AdjointLinearOperator<data_t>::applyImpl(const DataContainer<data_t>& y,
                                                  DataContainer<data_t>& Aty) const
    {
        if (op_->getRangeDescriptor().getNumberOfCoefficients() != y.getSize()
            || op_->getDomainDescriptor().getNumberOfCoefficients() != Aty.getSize()) {
            throw InvalidArgumentError(
                "AdjointLinearOperator::apply: incorrect input/output sizes for adjoint leaf");
        }

        op_->applyAdjoint(y, Aty);
    }

    template <typename data_t>
    void AdjointLinearOperator<data_t>::applyAdjointImpl(const DataContainer<data_t>& x,
                                                         DataContainer<data_t>& Ax) const
    {
        if (op_->getDomainDescriptor().getNumberOfCoefficients() != x.getSize()
            || op_->getRangeDescriptor().getNumberOfCoefficients() != Ax.getSize())
            throw InvalidArgumentError("LinearOperator::applyAdjoint: incorrect "
                                       "input/output sizes for adjoint leaf");

        op_->apply(x, Ax);
    }

    template <typename data_t>
    AdjointLinearOperator<data_t>* AdjointLinearOperator<data_t>::cloneImpl() const
    {
        return new AdjointLinearOperator<data_t>(*op_);
    }

    template <typename data_t>
    bool AdjointLinearOperator<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {

        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        auto otherAdjoint = downcast_safe<AdjointLinearOperator>(&other);
        if (!otherAdjoint)
            return false;

        return *op_ == *otherAdjoint->op_;
    }

    // ------------------------------------------
    // Implementation ScalarMulLinearOperator
    template <typename data_t>
    ScalarMulLinearOperator<data_t>::ScalarMulLinearOperator(data_t scalar,
                                                             const LinearOperator<data_t>& op)
        : LinearOperator<data_t>(op.getDomainDescriptor(), op.getRangeDescriptor()),
          scalar_(scalar),
          op_(op.clone())
    {
    }

    template <typename data_t>
    void ScalarMulLinearOperator<data_t>::applyImpl(const DataContainer<data_t>& x,
                                                    DataContainer<data_t>& Ax) const
    {
        // sanity check the arguments for the intended evaluation tree leaf operation
        if (op_->getDomainDescriptor().getNumberOfCoefficients() != x.getSize())
            throw InvalidArgumentError("ScalarMulLinearOperator::apply: incorrect input/output "
                                       "sizes for scalar mult. leaf");

        op_->apply(x, Ax);
        Ax *= scalar_;
    }

    template <typename data_t>
    void ScalarMulLinearOperator<data_t>::applyAdjointImpl(const DataContainer<data_t>& y,
                                                           DataContainer<data_t>& Aty) const
    {
        if (op_->getRangeDescriptor().getNumberOfCoefficients() != y.getSize())
            throw InvalidArgumentError("ScalarMulLinearOperator::apply: incorrect input/output "
                                       "sizes for scalar mult. leaf");

        op_->applyAdjoint(y, Aty);
        Aty *= scalar_;
    }

    template <typename data_t>
    ScalarMulLinearOperator<data_t>* ScalarMulLinearOperator<data_t>::cloneImpl() const
    {
        return new ScalarMulLinearOperator<data_t>(scalar_, *op_);
    }

    template <typename data_t>
    bool ScalarMulLinearOperator<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {

        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        auto otherdowncast = downcast_safe<ScalarMulLinearOperator>(&other);
        if (!otherdowncast)
            return false;

        return scalar_ == otherdowncast->scalar_ && *op_ == *otherdowncast->op_;
    }

    // ------------------------------------------
    // Implementation CompositeAddLinearOperator
    template <typename data_t>
    CompositeAddLinearOperator<data_t>::CompositeAddLinearOperator(
        const LinearOperator<data_t>& lhs, const LinearOperator<data_t>& rhs)
        : LinearOperator<data_t>(lhs.getDomainDescriptor(), rhs.getRangeDescriptor()),
          lhs_(lhs.clone()),
          rhs_(rhs.clone())
    {
    }

    template <typename data_t>
    void CompositeAddLinearOperator<data_t>::applyImpl(const DataContainer<data_t>& x,
                                                       DataContainer<data_t>& Ax) const
    {
        // sanity check the arguments for the intended evaluation tree leaf operation
        if (rhs_->getDomainDescriptor().getNumberOfCoefficients() != x.getSize()
            || rhs_->getRangeDescriptor().getNumberOfCoefficients() != Ax.getSize()
            || lhs_->getDomainDescriptor().getNumberOfCoefficients() != x.getSize()
            || lhs_->getRangeDescriptor().getNumberOfCoefficients() != Ax.getSize())
            throw InvalidArgumentError(
                "CompositeAddLinearOperator::apply: incorrect input/output sizes for add leaf");

        rhs_->apply(x, Ax);
        Ax += lhs_->apply(x);
    }

    template <typename data_t>
    void CompositeAddLinearOperator<data_t>::applyAdjointImpl(const DataContainer<data_t>& y,
                                                              DataContainer<data_t>& Aty) const
    {
        // sanity check the arguments for the intended evaluation tree leaf operation
        if (rhs_->getRangeDescriptor().getNumberOfCoefficients() != y.getSize()
            || rhs_->getDomainDescriptor().getNumberOfCoefficients() != Aty.getSize()
            || lhs_->getRangeDescriptor().getNumberOfCoefficients() != y.getSize()
            || lhs_->getDomainDescriptor().getNumberOfCoefficients() != Aty.getSize())
            throw InvalidArgumentError("CompositeAddLinearOperator::applyAdjoint: incorrect "
                                       "input/output sizes for add leaf");

        rhs_->applyAdjoint(y, Aty);
        Aty += lhs_->applyAdjoint(y);
    }

    template <typename data_t>
    CompositeAddLinearOperator<data_t>* CompositeAddLinearOperator<data_t>::cloneImpl() const
    {
        return new CompositeAddLinearOperator<data_t>(*lhs_, *rhs_);
    }

    template <typename data_t>
    bool CompositeAddLinearOperator<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        auto otherdowncast = downcast_safe<CompositeAddLinearOperator<data_t>>(&other);
        if (!otherdowncast)
            return false;

        return *lhs_ == *otherdowncast->lhs_ && *rhs_ == *otherdowncast->rhs_;
    }

    // ------------------------------------------
    // Implementation CompositeMulLinearOperator
    template <typename data_t>
    CompositeMulLinearOperator<data_t>::CompositeMulLinearOperator(
        const LinearOperator<data_t>& lhs, const LinearOperator<data_t>& rhs)
        : LinearOperator<data_t>(rhs.getDomainDescriptor(), lhs.getRangeDescriptor()),
          lhs_(lhs.clone()),
          rhs_(rhs.clone())
    {
    }

    template <typename data_t>
    void CompositeMulLinearOperator<data_t>::applyImpl(const DataContainer<data_t>& x,
                                                       DataContainer<data_t>& Ax) const
    {
        // sanity check the arguments for the intended evaluation tree leaf operation
        if (rhs_->getDomainDescriptor().getNumberOfCoefficients() != x.getSize()) {
            throw Error("CompositeMulLinearOperator::apply:{}: expected differently sized input x "
                        "(is {}, expected {})",
                        __LINE__, rhs_->getDomainDescriptor().getNumberOfCoefficients(),
                        x.getSize());
        }

        if (lhs_->getRangeDescriptor().getNumberOfCoefficients() != Ax.getSize()) {
            throw Error("CompositeMulLinearOperator::apply: expected differently sized input Ax "
                        "(is {}, expected {})",
                        lhs_->getRangeDescriptor().getNumberOfCoefficients(), Ax.getSize());
        }

        DataContainer<data_t> temp(rhs_->getRangeDescriptor(), x.getDataHandlerType());
        rhs_->apply(x, temp);
        lhs_->apply(temp, Ax);
    }

    template <typename data_t>
    void CompositeMulLinearOperator<data_t>::applyAdjointImpl(const DataContainer<data_t>& y,
                                                              DataContainer<data_t>& Aty) const
    {
        // sanity check the arguments for the intended evaluation tree leaf operation
        if (lhs_->getRangeDescriptor().getNumberOfCoefficients() != y.getSize()
            || rhs_->getDomainDescriptor().getNumberOfCoefficients() != Aty.getSize())
            throw InvalidArgumentError("CompositeMulLinearOperator::applyAdjoint: incorrect "
                                       "input/output sizes for mult leaf");

        DataContainer<data_t> temp(lhs_->getDomainDescriptor(), y.getDataHandlerType());
        lhs_->applyAdjoint(y, temp);
        rhs_->applyAdjoint(temp, Aty);
    }

    template <typename data_t>
    CompositeMulLinearOperator<data_t>* CompositeMulLinearOperator<data_t>::cloneImpl() const
    {
        return new CompositeMulLinearOperator<data_t>(*lhs_, *rhs_);
    }

    template <typename data_t>
    bool CompositeMulLinearOperator<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        auto otherdowncast = downcast_safe<CompositeMulLinearOperator<data_t>>(&other);
        if (!otherdowncast)
            return false;

        return *lhs_ == *otherdowncast->lhs_ && *rhs_ == *otherdowncast->rhs_;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class LinearOperator<float>;
    template class LinearOperator<complex<float>>;
    template class LinearOperator<double>;
    template class LinearOperator<complex<double>>;

    template class AdjointLinearOperator<float>;
    template class AdjointLinearOperator<complex<float>>;
    template class AdjointLinearOperator<double>;
    template class AdjointLinearOperator<complex<double>>;

    template class ScalarMulLinearOperator<float>;
    template class ScalarMulLinearOperator<complex<float>>;
    template class ScalarMulLinearOperator<double>;
    template class ScalarMulLinearOperator<complex<double>>;

    template class CompositeAddLinearOperator<float>;
    template class CompositeAddLinearOperator<complex<float>>;
    template class CompositeAddLinearOperator<double>;
    template class CompositeAddLinearOperator<complex<double>>;

    template class CompositeMulLinearOperator<float>;
    template class CompositeMulLinearOperator<complex<float>>;
    template class CompositeMulLinearOperator<double>;
    template class CompositeMulLinearOperator<complex<double>>;

} // namespace elsa
