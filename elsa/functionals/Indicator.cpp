#include "Indicator.h"

namespace elsa
{
    template <typename data_t>
    data_t Indicator<data_t>::evaluateImpl(const DataContainer<data_t>& Rx)
    {
        if (std::any_of(std::begin(Rx), std::end(Rx),
                        [this](auto&& x) { return constraintIsSatisfied(x); })) {
            return std::numeric_limits<data_t>::infinity();
        } else {
            return static_cast<data_t>(0);
        }
    }

    template <typename data_t>
    void Indicator<data_t>::getGradientInPlaceImpl(DataContainer<data_t>&)
    {
        throw LogicError("Indicator: not differentiable, so no gradient! (busted!)");
    }

    template <typename data_t>
    LinearOperator<data_t> Indicator<data_t>::getHessianImpl(const DataContainer<data_t>&)
    {
        throw LogicError("Indicator: not differentiable, so no Hessian! (busted!)");
    }

    template <typename data_t>
    Indicator<data_t>* Indicator<data_t>::cloneImpl() const
    {
        return new Indicator(this->getDomainDescriptor(), _comparator, _constraintValue);
    }

    template <typename data_t>
    bool Indicator<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (!Functional<data_t>::isEqual(other))
            return false;

        auto otherIndicator = dynamic_cast<const Indicator*>(&other);
        return static_cast<bool>(otherIndicator);
    }

    template <typename data_t>
    bool Indicator<data_t>::constraintIsSatisfied(data_t value)
    {
        return _comparator(value, _constraintValue);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class Indicator<float>;
    template class Indicator<elsa::complex<float>>;
    template class Indicator<double>;
    template class Indicator<elsa::complex<double>>;
} // namespace elsa
