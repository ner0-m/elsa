#include "HardThresholding.h"
#include "DataContainer.h"
#include "Error.h"
#include "TypeCasts.hpp"

namespace elsa
{
    template <typename data_t>
    DataContainer<data_t> HardThresholding<data_t>::apply(const DataContainer<data_t>& v,
                                                          geometry::Threshold<data_t> t) const
    {
        DataContainer<data_t> out{v.getDataDescriptor()};
        apply(v, t, out);
        return out;
    }

    template <typename data_t>
    void HardThresholding<data_t>::apply(const DataContainer<data_t>& v,
                                         geometry::Threshold<data_t> t,
                                         DataContainer<data_t>& prox) const
    {
        if (v.getSize() != prox.getSize()) {
            throw LogicError("HardThresholding: sizes of v and prox must match");
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

    // ------------------------------------------
    // explicit template instantiation
    template class HardThresholding<float>;
    template class HardThresholding<double>;
} // namespace elsa
