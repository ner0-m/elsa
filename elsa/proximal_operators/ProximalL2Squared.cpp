#include "ProximalL2Squared.h"
#include "DataContainer.h"

namespace elsa
{
    template <class data_t>
    ProximalL2Squared<data_t>::ProximalL2Squared(data_t sigma) : sigma_(sigma)
    {
    }

    template <class data_t>
    ProximalL2Squared<data_t>::ProximalL2Squared(const DataContainer<data_t>& b) : b_(b)
    {
    }

    template <class data_t>
    ProximalL2Squared<data_t>::ProximalL2Squared(const DataContainer<data_t>& b,
                                                 SelfType_t<data_t> sigma)
        : sigma_(sigma), b_(b)
    {
    }

    template <class data_t>
    DataContainer<data_t> ProximalL2Squared<data_t>::apply(const DataContainer<data_t>& v,
                                                           SelfType_t<data_t> t) const
    {
        auto out = DataContainer<data_t>(v.getDataDescriptor());
        apply(v, t, out);
        return out;
    }

    template <class data_t>
    void ProximalL2Squared<data_t>::apply(const DataContainer<data_t>& v, SelfType_t<data_t> lambda,
                                          DataContainer<data_t>& prox) const
    {
        const auto denom = data_t{1} + (lambda * sigma_);

        if (b_.has_value()) {
            lincomb(1 / denom, v, (sigma_ * lambda) / denom, *b_, prox);
        } else {
            prox = v / denom;
        }
    }

    // ------------------------------------------
    // explicit template instantiation
    template class ProximalL2Squared<float>;
    template class ProximalL2Squared<double>;
} // namespace elsa
