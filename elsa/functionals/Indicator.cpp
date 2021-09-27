#include "Indicator.h"

namespace elsa
{
    template <typename data_t>
    Indicator<data_t>::Indicator(const DataDescriptor& domainDescriptor)
        : Functional<data_t>(domainDescriptor),
          _constraintOperation{ComparisonOperation::GREATER_EQUAL_THAN},
          _constraintValue{0}
    {
    }

    template <typename data_t>
    Indicator<data_t>::Indicator(const DataDescriptor& domainDescriptor,
                                 ComparisonOperation constraintOperation, data_t constraintValue)
        : Functional<data_t>(domainDescriptor),
          _constraintOperation{constraintOperation},
          _constraintValue{constraintValue}
    {
    }

    template <typename data_t>
    data_t Indicator<data_t>::evaluateImpl(const DataContainer<data_t>& Rx)
    {
        for (data_t val : Rx) {
            if (constraintIsSatisfied(val)) {
                return std::numeric_limits<data_t>::infinity();
            }
        }
        return static_cast<data_t>(0);
    }

    template <typename data_t>
    void Indicator<data_t>::getGradientInPlaceImpl([[maybe_unused]] DataContainer<data_t>& Rx)
    {
        throw LogicError("Indicator: not differentiable, so no gradient! (busted!)");
    }

    template <typename data_t>
    LinearOperator<data_t> Indicator<data_t>::getHessianImpl(const DataContainer<data_t>& Rx)
    {
        throw LogicError("Indicator: not differentiable, so no Hessian! (busted!)");
    }

    template <typename data_t>
    Indicator<data_t>* Indicator<data_t>::cloneImpl() const
    {
        return new Indicator(this->getDomainDescriptor(), _constraintOperation, _constraintValue);
    }

    template <typename data_t>
    bool Indicator<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (!Functional<data_t>::isEqual(other))
            return false;

        auto otherIndicator = dynamic_cast<const Indicator*>(&other);
        return static_cast<bool>(otherIndicator);
    }

    // TODO this can be improved using lambdas and/or functionals
    template <typename data_t>
    bool Indicator<data_t>::constraintIsSatisfied(data_t value)
    {
        if (_constraintOperation == ComparisonOperation::EQUAL_TO) {
            return !(value == _constraintValue);
        } else if (_constraintOperation == ComparisonOperation::NOT_EQUAL_TO) {
            return !(value != _constraintValue);
        } else if (_constraintOperation == ComparisonOperation::GREATER_THAN) {
            return !(value > _constraintValue);
        } else if (_constraintOperation == ComparisonOperation::LESS_THAN) {
            return !(value < _constraintValue);
        } else if (_constraintOperation == ComparisonOperation::GREATER_EQUAL_THAN) {
            return !(value >= _constraintValue);
        } else if (_constraintOperation == ComparisonOperation::LESS_EQUAL_THAN) {
            return !(value <= _constraintValue);
        }
    }

    // ------------------------------------------
    // explicit template instantiation
    template class Indicator<float>;
    template class Indicator<double>;
} // namespace elsa
