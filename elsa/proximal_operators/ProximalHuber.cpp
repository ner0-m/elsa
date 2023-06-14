#include "ProximalHuber.h"
#include "DataContainer.h"

namespace elsa
{
    template <class data_t>
    ProximalHuber<data_t>::ProximalHuber(data_t delta) : delta_(delta)
    {
    }

    template <class data_t>
    DataContainer<data_t> ProximalHuber<data_t>::apply(const DataContainer<data_t>& v,
                                                       SelfType_t<data_t> t) const
    {
        DataContainer<data_t> out{v.getDataDescriptor()};
        apply(v, t, out);
        return out;
    }

    template <class data_t>
    void ProximalHuber<data_t>::apply(const DataContainer<data_t>& v, SelfType_t<data_t> t,
                                      DataContainer<data_t>& prox) const
    {
        auto factor = 1 - (t / (std::max(v.l2Norm(), t) + delta_));
        prox = factor * v;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class ProximalHuber<float>;
    template class ProximalHuber<double>;
} // namespace elsa
