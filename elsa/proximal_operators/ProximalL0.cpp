#include "ProximalL0.h"
#include "DataContainer.h"
#include "Error.h"
#include "TypeCasts.hpp"

namespace elsa
{
    template <typename data_t>
    DataContainer<data_t> ProximalL0<data_t>::apply(const DataContainer<data_t>& v,
                                                    geometry::Threshold<data_t> t) const
    {
        DataContainer<data_t> out{v.getDataDescriptor()};
        apply(v, t, out);
        return out;
    }

    template <typename data_t>
    void ProximalL0<data_t>::apply(const DataContainer<data_t>& v, geometry::Threshold<data_t> t,
                                   DataContainer<data_t>& prox) const
    {
        if (v.getSize() != prox.getSize()) {
            throw LogicError("ProximalL0: sizes of v and prox must match");
        }

        auto vIter = v.begin();
        auto proxIter = prox.begin();

        for (; vIter != v.end() && proxIter != prox.end(); vIter++, proxIter++) {
            if ((*vIter > t) || (*vIter < -t)) {
                *proxIter = *vIter;
            } else {
                *proxIter = 0;
            }
        }
    }

    template <typename data_t>
    bool operator==(const ProximalL0<data_t>&, const ProximalL0<data_t>&)
    {
        return true;
    }

    template <typename data_t>
    bool operator!=(const ProximalL0<data_t>&, const ProximalL0<data_t>&)
    {
        return false;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class ProximalL0<float>;
    template class ProximalL0<double>;
} // namespace elsa
