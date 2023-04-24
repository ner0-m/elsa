#include <iostream>
#include "ProximalBoxConstraint.h"
#include "DataContainer.h"

namespace elsa
{

    template <class data_t>
    ProximalBoxConstraint<data_t>::ProximalBoxConstraint(data_t lower) : lower_(lower)
    {
    }

    template <class data_t>
    ProximalBoxConstraint<data_t>::ProximalBoxConstraint(data_t lower, data_t upper)
        : lower_(lower), upper_(upper)
    {
    }

    template <class data_t>
    DataContainer<data_t> ProximalBoxConstraint<data_t>::apply(const DataContainer<data_t>& v,
                                                               SelfType_t<data_t> t) const
    {
        DataContainer<data_t> out{v.getDataDescriptor()};
        apply(v, t, out);
        return out;
    }

    template <class data_t>
    void ProximalBoxConstraint<data_t>::apply(const DataContainer<data_t>& v, SelfType_t<data_t>,
                                              DataContainer<data_t>& prox) const
    {
        if (lower_.has_value() && upper_.has_value()) {
            prox = ::elsa::clip(v, *lower_, *upper_);
        } else if (lower_.has_value() && !upper_.has_value()) {
            prox = ::elsa::maximum(v, *lower_);
        } else if (!lower_.has_value() && upper_.has_value()) {
            prox = ::elsa::minimum(v, *upper_);
        } else {
            prox = v;
        }
    }

    template <class data_t>
    bool operator==(const ProximalBoxConstraint<data_t>& lhs,
                    const ProximalBoxConstraint<data_t>& rhs)
    {
        return lhs.lower_.has_value() && rhs.lower_.has_value() && lhs.upper_.has_value()
               && rhs.upper_.has_value() && (*lhs.lower) == (rhs.lower_)
               && (*lhs.upper_) == (*rhs.upper_);
    }

    template <class data_t>
    bool operator!=(const ProximalBoxConstraint<data_t>& lhs,
                    const ProximalBoxConstraint<data_t>& rhs)
    {
        return !(lhs == rhs);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class ProximalBoxConstraint<float>;
    template class ProximalBoxConstraint<double>;
} // namespace elsa
