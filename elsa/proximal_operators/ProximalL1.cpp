#include <cmath>

#include "ProximalL1.h"
#include "DataContainer.h"
#include "Error.h"
#include "TypeCasts.hpp"
#include "Math.hpp"
#include "elsaDefines.h"

namespace elsa
{
    template <typename data_t>
    ProximalL1<data_t>::ProximalL1(data_t sigma) : sigma_(sigma)
    {
    }

    template <typename data_t>
    DataContainer<data_t> ProximalL1<data_t>::apply(const DataContainer<data_t>& v,
                                                    SelfType_t<data_t> t) const
    {
        DataContainer<data_t> out{v.getDataDescriptor()};
        apply(v, t, out);
        return out;
    }

    template <typename data_t>
    void ProximalL1<data_t>::apply(const DataContainer<data_t>& v, SelfType_t<data_t> t,
                                   DataContainer<data_t>& prox) const
    {
        if (v.getSize() != prox.getSize()) {
            throw LogicError("ProximalL1: sizes of v and prox must match");
        }

        auto first = v.begin();
        auto out = prox.begin();

        for (; first != v.end() && out != prox.end(); first++, out++) {
            auto tmp = std::abs(*first) - (t * sigma_);
            tmp = 0.5 * (tmp + std::abs(tmp));

            *out = sign(*first) * tmp;
        }
    }

    // ------------------------------------------
    // explicit template instantiation
    template class ProximalL1<float>;
    template class ProximalL1<double>;
} // namespace elsa
