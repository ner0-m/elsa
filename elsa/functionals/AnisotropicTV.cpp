#include "AnisotropicTV.h"
#include "FiniteDifferences.h"

namespace elsa
{
    template <typename data_t>
    AnisotropicTV<data_t>::AnisotropicTV(const DataDescriptor& domainDescriptor)
        : Functional<data_t>(domainDescriptor)
    {
    }

    template <typename data_t>
    data_t AnisotropicTV<data_t>::evaluateImpl(const DataContainer<data_t>& Rx)
    {
        FiniteDifferences<data_t> fdOperator{Rx.getDataDescriptor()};
        auto gradient = fdOperator.apply(Rx);
        return gradient.l1Norm();
    }

    template <typename data_t>
    void AnisotropicTV<data_t>::getGradientImpl(const DataContainer<data_t>& Rx,
                                                DataContainer<data_t>& out)
    {
        throw LogicError("IsotropicTV: not differentiable, so no gradient! (busted!)");
    }
    template <typename data_t>
    LinearOperator<data_t> AnisotropicTV<data_t>::getHessianImpl(const DataContainer<data_t>& Rx)
    {
        throw LogicError("IsotropicTV: not differentiable, so no hessian! (busted!)");
    }
    template <typename data_t>
    AnisotropicTV<data_t>* AnisotropicTV<data_t>::cloneImpl() const
    {
        return new AnisotropicTV(this->getDomainDescriptor());
    }
    template <typename data_t>
    bool AnisotropicTV<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (!Functional<data_t>::isEqual(other))
            return false;

        return is<AnisotropicTV>(other);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class AnisotropicTV<float>;
    template class AnisotropicTV<double>;
    template class AnisotropicTV<complex<float>>;
    template class AnisotropicTV<complex<double>>;

} // namespace elsa