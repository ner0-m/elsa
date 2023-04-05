#include "WeightedL2Squared.h"
#include "DataContainer.h"
#include "LinearOperator.h"
#include "Scaling.h"
#include "TypeCasts.hpp"

#include <stdexcept>

namespace elsa
{
    template <typename data_t>
    WeightedL2Squared<data_t>::WeightedL2Squared(const DataContainer<data_t>& weights)
        : Functional<data_t>(weights.getDataDescriptor()), weights_{weights}
    {
    }

    template <typename data_t>
    bool WeightedL2Squared<data_t>::isDifferentiable() const
    {
        return true;
    }

    template <typename data_t>
    Scaling<data_t> WeightedL2Squared<data_t>::getWeightingOperator() const
    {
        return Scaling<data_t>(weights_.getDataDescriptor(), weights_);
    }

    template <typename data_t>
    data_t WeightedL2Squared<data_t>::evaluateImpl(const DataContainer<data_t>& Rx)
    {
        auto temp = weights_ * Rx;
        return static_cast<data_t>(0.5) * Rx.dot(temp);
    }

    template <typename data_t>
    void WeightedL2Squared<data_t>::getGradientImpl(const DataContainer<data_t>& Rx,
                                                    DataContainer<data_t>& out)
    {
        out = weights_ * Rx;
    }

    template <typename data_t>
    LinearOperator<data_t> WeightedL2Squared<data_t>::getHessianImpl(const DataContainer<data_t>&)
    {
        return leaf(getWeightingOperator());
    }

    template <typename data_t>
    WeightedL2Squared<data_t>* WeightedL2Squared<data_t>::cloneImpl() const
    {
        return new WeightedL2Squared(weights_);
    }

    template <typename data_t>
    bool WeightedL2Squared<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (!Functional<data_t>::isEqual(other))
            return false;

        auto otherWL2 = downcast_safe<WeightedL2Squared>(&other);
        if (!otherWL2)
            return false;

        return weights_ == otherWL2->weights_;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class WeightedL2Squared<float>;
    template class WeightedL2Squared<double>;
    template class WeightedL2Squared<complex<float>>;
    template class WeightedL2Squared<complex<double>>;

} // namespace elsa
