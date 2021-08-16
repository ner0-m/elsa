#include "Constraint.h"

namespace elsa
{
    template <typename data_t>
    Constraint<data_t>::Constraint(const LinearOperator<data_t>& A, const LinearOperator<data_t>& B,
                                   const DataContainer<data_t>& c)
        : _A{A.clone()}, _B{B.clone()}, _c{c}
    {
    }

    template <typename data_t>
    auto Constraint<data_t>::getOperatorA() const -> const LinearOperator<data_t>&
    {
        return *_A;
    }

    template <typename data_t>
    auto Constraint<data_t>::getOperatorB() const -> const LinearOperator<data_t>&
    {
        return *_B;
    }

    template <typename data_t>
    auto Constraint<data_t>::getDataVectorC() const -> const DataContainer<data_t>&
    {
        return _c;
    }

    template <typename data_t>
    auto Constraint<data_t>::cloneImpl() const -> Constraint<data_t>*
    {
        return new Constraint<data_t>(*_A, *_B, _c);
    }

    template <typename data_t>
    auto Constraint<data_t>::isEqual(const Constraint<data_t>& other) const -> bool
    {
        if (other.getOperatorA() != *_A) {
            return false;
        }

        if (other.getOperatorB() != *_B) {
            return false;
        }

        if (other.getDataVectorC() != _c) {
            return false;
        }

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class Constraint<float>;
    template class Constraint<std::complex<float>>;
    template class Constraint<double>;
    template class Constraint<std::complex<double>>;
} // namespace elsa
