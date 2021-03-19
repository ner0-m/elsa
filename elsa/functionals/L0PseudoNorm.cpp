#include "L0PseudoNorm.h"

namespace elsa
{
    template <typename data_t>
    L0PseudoNorm<data_t>::L0PseudoNorm(const DataDescriptor& domainDescriptor)
        : Functional<data_t>(domainDescriptor)
    {
    }

    template <typename data_t>
    L0PseudoNorm<data_t>::L0PseudoNorm(const Residual<data_t>& residual)
        : Functional<data_t>(residual)
    {
    }

    template <typename data_t>
    auto L0PseudoNorm<data_t>::evaluateImpl(const DataContainer<data_t>& Rx) -> data_t
    {
        return static_cast<data_t>(Rx.l0PseudoNorm());
    }

    template <typename data_t>
    void L0PseudoNorm<data_t>::getGradientInPlaceImpl([[maybe_unused]] DataContainer<data_t>& Rx)
    {
        throw std::logic_error("L0PseudoNorm: not differentiable, so no gradient! (busted!)");
    }

    template <typename data_t>
    auto L0PseudoNorm<data_t>::getHessianImpl([[maybe_unused]] const DataContainer<data_t>& Rx)
        -> LinearOperator<data_t>
    {
        throw std::logic_error("L0PseudoNorm: not differentiable, so no Hessian! (busted!)");
    }

    template <typename data_t>
    auto L0PseudoNorm<data_t>::cloneImpl() const -> L0PseudoNorm<data_t>*
    {
        return new L0PseudoNorm(this->getResidual());
    }

    template <typename data_t>
    auto L0PseudoNorm<data_t>::isEqual(const Functional<data_t>& other) const -> bool
    {
        if (!Functional<data_t>::isEqual(other))
            return false;

        auto otherL0PseudoNorm = dynamic_cast<const L0PseudoNorm*>(&other);
        return static_cast<bool>(otherL0PseudoNorm);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class L0PseudoNorm<float>;
    template class L0PseudoNorm<double>;
} // namespace elsa
