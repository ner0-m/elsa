#include <cmath>

#include "ProximalL1.h"
#include "DataContainer.h"
#include "Error.h"
#include "TypeCasts.hpp"
#include "Math.hpp"

namespace elsa
{

    template <typename data_t>
    DataContainer<data_t> ProximalL1<data_t>::apply(const DataContainer<data_t>& v,
                                                    geometry::Threshold<data_t> t) const
    {
        DataContainer<data_t> out{v.getDataDescriptor()};
        apply(v, t, out);
        return out;
    }

    template <typename data_t>
    void ProximalL1<data_t>::apply(const DataContainer<data_t>& v, geometry::Threshold<data_t> t,
                                   DataContainer<data_t>& prox) const
    {
        if (v.getSize() != prox.getSize()) {
            throw LogicError("ProximalL1: sizes of v and prox must match");
        }

        auto first = v.begin();
        auto out = prox.begin();

        for (; first != v.end() && out != prox.end(); first++, out++) {
            *out = std::max(std::abs(*first) - t, data_t{0}) * sign<data_t, data_t>(*first);
        }
    }

    // ------------------------------------------
    // explicit template instantiation
    template class ProximalL1<float>;
    template class ProximalL1<double>;
} // namespace elsa
