#include "IsotropicTV.h"
#include "FiniteDifferences.h"

namespace elsa
{
    template <typename data_t>
    IsotropicTV<data_t>::IsotropicTV(const DataDescriptor& domainDescriptor)
        : Functional<data_t>(domainDescriptor)
    {
    }

    template <typename data_t>
    data_t IsotropicTV<data_t>::evaluateImpl(const DataContainer<data_t>& Rx)
    {
        FiniteDifferences<data_t> fdOperator(Rx.getDataDescriptor());
        auto gradient = fdOperator.apply(Rx);
        return gradient.l12MixedNorm();
    }

    template <typename data_t>
    void IsotropicTV<data_t>::getGradientImpl(const DataContainer<data_t>& Rx,
                                              DataContainer<data_t>& out)
    {
        throw LogicError("IsotropicTV: not differentiable, so no gradient! (busted!)");
    }
    template <typename data_t>
    LinearOperator<data_t> IsotropicTV<data_t>::getHessianImpl(const DataContainer<data_t>& Rx)
    {
        throw LogicError("IsotropicTV: not differentiable, so no hessian! (busted!)");
    }
    template <typename data_t>
    IsotropicTV<data_t>* IsotropicTV<data_t>::cloneImpl() const
    {
        return new IsotropicTV(this->getDomainDescriptor());
    }
    template <typename data_t>
    bool IsotropicTV<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (!Functional<data_t>::isEqual(other))
            return false;

        return is<IsotropicTV>(other);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class IsotropicTV<float>;
    template class IsotropicTV<double>;
    template class IsotropicTV<complex<float>>;
    template class IsotropicTV<complex<double>>;

} // namespace elsa