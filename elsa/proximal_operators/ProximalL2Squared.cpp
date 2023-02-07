#include "ProximalL2Squared.h"

namespace elsa
{
    template <class data_t>
    ProximalL2Squared<data_t>::ProximalL2Squared(const DataContainer<data_t>& b) : b_(b)
    {
    }

    template <class data_t>
    DataContainer<data_t> ProximalL2Squared<data_t>::apply(const DataContainer<data_t>& v,
                                                           geometry::Threshold<data_t> t) const
    {
        auto out = DataContainer<data_t>(v.getDataDescriptor());
        apply(v, t, out);
        return out;
    }

    template <class data_t>
    void ProximalL2Squared<data_t>::apply(const DataContainer<data_t>& v,
                                          geometry::Threshold<data_t> t,
                                          DataContainer<data_t>& prox) const
    {
        const auto f1 = data_t{1} + data_t{2} * data_t{t};
        prox = v;

        if (b_.has_value()) {
            const auto f2 = data_t{1} / f1;
            const auto f3 = data_t{2} * data_t{t} / f1;

            prox *= f2;
            prox += (*b_) * f3;
        } else {
            prox *= f1;
        }
    }

    // ------------------------------------------
    // explicit template instantiation
    template class ProximalL2Squared<float>;
    template class ProximalL2Squared<double>;
} // namespace elsa
