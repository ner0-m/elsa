#include "ProximalMixedL21Norm.h"
#include "DataContainer.h"
#include "BlockDescriptor.h"
#include "spdlog/fmt/bundled/core.h"

namespace elsa
{
    template <class data_t>
    ProximalMixedL21Norm<data_t>::ProximalMixedL21Norm(data_t sigma) : sigma_(sigma)
    {
    }

    template <class data_t>
    DataContainer<data_t> ProximalMixedL21Norm<data_t>::apply(const DataContainer<data_t>& v,
                                                              SelfType_t<data_t> t) const
    {
        DataContainer<data_t> out{v.getDataDescriptor()};
        apply(v, t, out);
        return out;
    }

    template <class data_t>
    void ProximalMixedL21Norm<data_t>::apply(const DataContainer<data_t>& v, SelfType_t<data_t> t,
                                             DataContainer<data_t>& prox) const
    {
        if (!is<BlockDescriptor>(v.getDataDescriptor())) {
            throw Error("ProximalMixedL21Norm: Blocked DataContainer expected");
        }

        auto p21norm = v.pL2Norm();

        // set each block of prox to be tmp
        for (int i = 0; i < v.getNumberOfBlocks(); ++i) {
            prox.getBlock(i) = p21norm;
        }

        auto tau = t * sigma_;
        prox = (1 - tau / maximum(prox, tau)) * v;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class ProximalMixedL21Norm<float>;
    template class ProximalMixedL21Norm<double>;
} // namespace elsa
