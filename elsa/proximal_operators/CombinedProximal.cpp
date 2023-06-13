#include "CombinedProximal.h"

#include "BlockDescriptor.h"
#include "DataContainer.h"
#include "Error.h"
#include "ProximalOperator.h"
#include "TypeCasts.hpp"
#include "StrongTypes.h"
#include "elsaDefines.h"

namespace elsa
{
    template <class data_t>
    CombinedProximal<data_t>::CombinedProximal(ProximalOperator<data_t> prox) : proxs_({prox})
    {
    }

    template <class data_t>
    CombinedProximal<data_t>::CombinedProximal(ProximalOperator<data_t> prox1,
                                               ProximalOperator<data_t> prox2)
        : proxs_({prox1, prox2})
    {
    }

    template <class data_t>
    CombinedProximal<data_t>::CombinedProximal(ProximalOperator<data_t> prox1,
                                               ProximalOperator<data_t> prox2,
                                               ProximalOperator<data_t> prox3)
        : proxs_({prox1, prox2, prox3})
    {
    }

    template <class data_t>
    CombinedProximal<data_t>::CombinedProximal(ProximalOperator<data_t> prox1,
                                               ProximalOperator<data_t> prox2,
                                               ProximalOperator<data_t> prox3,
                                               ProximalOperator<data_t> prox4)
        : proxs_({prox1, prox2, prox3, prox4})
    {
    }

    template <class data_t>
    DataContainer<data_t> CombinedProximal<data_t>::apply(const DataContainer<data_t>& v,
                                                          SelfType_t<data_t> t) const
    {
        DataContainer<data_t> out(v.getDataDescriptor());
        apply(v, t, out);
        return out;
    }

    template <class data_t>
    void CombinedProximal<data_t>::apply(const DataContainer<data_t>& v, SelfType_t<data_t> t,
                                         DataContainer<data_t>& prox) const
    {
        if (!is<BlockDescriptor>(v.getDataDescriptor())) {
            throw Error("CombinedProximal: Proximal needs to be blocked");
        }

        auto& blockedDesc = downcast_safe<BlockDescriptor>(v.getDataDescriptor());

        if (blockedDesc.getNumberOfBlocks() != proxs_.size()) {
            throw Error("CombinedProximal: number of blocks ({}) and number of proximals ({}) "
                        "do not fit",
                        blockedDesc.getNumberOfBlocks(), proxs_.size());
        }

        for (int i = 0; i < blockedDesc.getNumberOfBlocks(); ++i) {
            auto outview = prox.getBlock(i);
            auto inview = v.getBlock(i);

            proxs_[i].apply(inview, t, outview);
        }
    }

    template <class data_t>
    ProximalOperator<data_t> CombinedProximal<data_t>::getIthProximal(index_t i)
    {
        return proxs_.at(i);
    }

    template <class data_t>
    void CombinedProximal<data_t>::addProximal(ProximalOperator<data_t> prox)
    {
        proxs_.push_back(prox);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class CombinedProximal<float>;
    template class CombinedProximal<double>;
} // namespace elsa
