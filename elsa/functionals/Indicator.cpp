#include "Indicator.h"

namespace elsa
{
    template <typename data_t>
    Indicator<data_t>::Indicator(const DataDescriptor& domainDescriptor)
        : Functional<data_t>(domainDescriptor)
    {
    }

    template <typename data_t>
    data_t Indicator<data_t>::evaluateImpl(const DataContainer<data_t>& Rx)
    {
        for (data_t val : Rx) {
            if (val < 0) {
                return std::numeric_limits<data_t>::infinity();
            }
        }
        return static_cast<data_t>(0);

        // return Rx is in the set S ? 0: nonZeroValue;
    }

    template <typename data_t>
    void Indicator<data_t>::getGradientInPlaceImpl([[maybe_unused]] DataContainer<data_t>& Rx)
    {
        // TODO
        throw LogicError("Indicator: not differentiable, so no gradient! (busted!)");
    }

    template <typename data_t>
    LinearOperator<data_t> Indicator<data_t>::getHessianImpl(const DataContainer<data_t>& Rx)
    {
        // TODO
        throw LogicError("Indicator: not differentiable, so no Hessian! (busted!)");
    }

    template <typename data_t>
    Indicator<data_t>* Indicator<data_t>::cloneImpl() const
    {
        return new Indicator(this->getDomainDescriptor());
    }

    template <typename data_t>
    bool Indicator<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (!Functional<data_t>::isEqual(other))
            return false;

        auto otherIndicator = dynamic_cast<const Indicator*>(&other);
        return static_cast<bool>(otherIndicator);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class Indicator<float>;
    template class Indicator<double>;
    // TODO
    // template class Indicator<std::complex<float>>;
    // template class Indicator<std::complex<double>>;
} // namespace elsa
