#include "Huber.h"
#include "Scaling.h"
#include "TypeCasts.hpp"

#include <cmath>
#include <stdexcept>

namespace elsa
{
    template <typename data_t>
    Huber<data_t>::Huber(const DataDescriptor& domainDescriptor, real_t delta)
        : Functional<data_t>(domainDescriptor), _delta{delta}
    {
        // sanity check delta
        if (delta <= static_cast<real_t>(0.0))
            throw InvalidArgumentError("Huber: delta has to be positive.");
    }

    template <typename data_t>
    Huber<data_t>::Huber(const elsa::Residual<data_t>& residual, real_t delta)
        : Functional<data_t>(residual), _delta{delta}
    {
        // sanity check delta
        if (delta <= static_cast<real_t>(0.0))
            throw InvalidArgumentError("Huber: delta has to be positive.");
    }

    template <typename data_t>
    data_t Huber<data_t>::evaluateImpl(const DataContainer<data_t>& Rx)
    {
        // note: this is currently not a reduction in DataContainer, but implemented here "manually"

        auto result = static_cast<data_t>(0.0);

        for (index_t i = 0; i < Rx.getSize(); ++i) {
            data_t value = Rx[i];
            if (std::abs(value) <= _delta)
                result += static_cast<data_t>(0.5) * value * value;
            else
                result += _delta * (std::abs(value) - static_cast<real_t>(0.5) * _delta);
        }

        return result;
    }

    template <typename data_t>
    void Huber<data_t>::getGradientInPlaceImpl(DataContainer<data_t>& Rx)
    {
        for (index_t i = 0; i < Rx.getSize(); ++i) {
            data_t value = Rx[i];
            if (value > _delta)
                Rx[i] = _delta;
            else if (value < -_delta)
                Rx[i] = -_delta;
            // else Rx[i] = Rx[i], i.e. nothing to do for the quadratic case
        }
    }

    template <typename data_t>
    std::unique_ptr<LinearOperator<data_t>>
        Huber<data_t>::getHessianImpl(const DataContainer<data_t>& Rx)
    {
        DataContainer<data_t> scaleFactors(Rx.getDataDescriptor());
        for (index_t i = 0; i < Rx.getSize(); ++i) {
            if (std::abs(Rx[i]) <= _delta)
                scaleFactors[i] = static_cast<data_t>(1);
            else
                scaleFactors[i] = static_cast<data_t>(0);
        }

        return std::make_unique<Scaling<data_t>>(Rx.getDataDescriptor(), scaleFactors);
    }

    template <typename data_t>
    Huber<data_t>* Huber<data_t>::cloneImpl() const
    {
        return new Huber(this->getResidual(), _delta);
    }

    template <typename data_t>
    bool Huber<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (!Functional<data_t>::isEqual(other))
            return false;

        auto otherHuber = downcast_safe<Huber>(&other);
        if (!otherHuber)
            return false;

        if (_delta != otherHuber->_delta)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class Huber<float>;
    template class Huber<double>;
    // no complex-number instantiations for Huber! (they would not really be useful)

} // namespace elsa
