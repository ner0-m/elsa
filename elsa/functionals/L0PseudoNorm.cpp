#include "L0PseudoNorm.h"
#include "DataContainer.h"
#include "TypeCasts.hpp"

namespace elsa
{
    template <typename data_t>
    L0PseudoNorm<data_t>::L0PseudoNorm(const DataDescriptor& domainDescriptor)
        : Functional<data_t>(domainDescriptor)
    {
    }

    template <typename data_t>
    data_t L0PseudoNorm<data_t>::evaluateImpl(const DataContainer<data_t>& Rx)
    {
        return static_cast<data_t>(Rx.l0PseudoNorm());
    }

    template <typename data_t>
    void L0PseudoNorm<data_t>::getGradientImpl(const DataContainer<data_t>&, DataContainer<data_t>&)
    {
        throw std::logic_error("L0PseudoNorm: not differentiable, so no gradient! (busted!)");
    }

    template <typename data_t>
    LinearOperator<data_t> L0PseudoNorm<data_t>::getHessianImpl(const DataContainer<data_t>&)
    {
        throw std::logic_error("L0PseudoNorm: not differentiable, so no Hessian! (busted!)");
    }

    template <typename data_t>
    L0PseudoNorm<data_t>* L0PseudoNorm<data_t>::cloneImpl() const
    {
        return new L0PseudoNorm(this->getDomainDescriptor());
    }

    template <typename data_t>
    bool L0PseudoNorm<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (!Functional<data_t>::isEqual(other))
            return false;

        return is<L0PseudoNorm>(other);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class L0PseudoNorm<float>;
    template class L0PseudoNorm<double>;
    template class L0PseudoNorm<complex<float>>;
    template class L0PseudoNorm<complex<double>>;
} // namespace elsa
