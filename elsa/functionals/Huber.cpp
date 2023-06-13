#include "Huber.h"
#include "DataContainer.h"
#include "Scaling.h"
#include "TypeCasts.hpp"

#include <cmath>
#include <stdexcept>

namespace elsa
{
    template <typename data_t>
    Huber<data_t>::Huber(const DataDescriptor& domainDescriptor, real_t delta)
        : Functional<data_t>(domainDescriptor), delta_{delta}
    {
        // sanity check delta
        if (delta <= static_cast<real_t>(0.0))
            throw InvalidArgumentError("Huber: delta has to be positive.");
    }

    template <typename data_t>
    bool Huber<data_t>::isDifferentiable() const
    {
        return true;
    }

    template <typename data_t>
    data_t Huber<data_t>::evaluateImpl(const DataContainer<data_t>& x)
    {
        // note: this is currently not a reduction in DataContainer, but implemented here "manually"
        auto result = data_t{0.0};

        for (index_t i = 0; i < x.getSize(); ++i) {
            data_t value = x[i];
            if (std::abs(value) <= delta_)
                result += static_cast<data_t>(0.5) * value * value;
            else
                result += delta_ * (std::abs(value) - static_cast<real_t>(0.5) * delta_);
        }

        return result;
    }

    template <typename data_t>
    void Huber<data_t>::getGradientImpl(const DataContainer<data_t>& x, DataContainer<data_t>& out)
    {
        for (index_t i = 0; i < x.getSize(); ++i) {
            data_t value = x[i];
            if (value > delta_) {
                out[i] = delta_;
            } else if (value < -delta_) {
                out[i] = -delta_;
            } else {
                out[i] = x[i];
            }
        }
    }

    template <typename data_t>
    LinearOperator<data_t> Huber<data_t>::getHessianImpl(const DataContainer<data_t>& x)
    {
        DataContainer<data_t> s(x.getDataDescriptor());
        for (index_t i = 0; i < x.getSize(); ++i) {
            if (std::abs(x[i]) <= delta_) {
                s[i] = data_t{1};
            } else {
                s[i] = data_t{0};
            }
        }

        return leaf(Scaling<data_t>(x.getDataDescriptor(), s));
    }

    template <typename data_t>
    Huber<data_t>* Huber<data_t>::cloneImpl() const
    {
        return new Huber(this->getDomainDescriptor(), delta_);
    }

    template <typename data_t>
    bool Huber<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (!Functional<data_t>::isEqual(other))
            return false;

        auto otherHuber = downcast_safe<Huber>(&other);
        if (!otherHuber)
            return false;

        if (delta_ != otherHuber->delta_)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class Huber<float>;
    template class Huber<double>;
    // no complex-number instantiations for Huber! (they would not really be useful)

} // namespace elsa
