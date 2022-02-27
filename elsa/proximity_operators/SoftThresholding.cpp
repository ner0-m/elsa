#include "SoftThresholding.h"
#include "Error.h"
#include "TypeCasts.hpp"

namespace elsa
{
    template <typename data_t>
    SoftThresholding<data_t>::SoftThresholding(const DataDescriptor& descriptor)
        : ProximityOperator<data_t>(descriptor)
    {
    }

    template <typename data_t>
    void SoftThresholding<data_t>::applyImpl(const DataContainer<data_t>& v,
                                             geometry::Threshold<data_t> t,
                                             DataContainer<data_t>& prox) const
    {
        if (v.getSize() != prox.getSize()) {
            throw LogicError("SoftThresholding: sizes of v and prox must match");
        }

        auto vIter = v.begin();
        auto proxIter = prox.begin();

        for (; vIter != v.end() && proxIter != prox.end(); vIter++, proxIter++) {
            if (*vIter > t) {
                *proxIter = *vIter - t;
            } else if (*vIter < -t) {
                *proxIter = *vIter + t;
            } else {
                *proxIter = 0;
            }
        }
    }

    template <typename data_t>
    void SoftThresholding<data_t>::applyImpl(const DataContainer<data_t>& v,
                                             std::vector<geometry::Threshold<data_t>> thresholds,
                                             DataContainer<data_t>& prox) const
    {
        if (v.getSize() != prox.getSize()) {
            throw LogicError("SoftThresholding: sizes of v and prox must match");
        }

        if (v.getSize() != thresholds.size()) {
            throw LogicError("SoftThresholding: sizes of v and thresholds must match");
        }

        auto vIter = v.begin();
        auto thresholdsIter = thresholds.begin();
        auto proxIter = prox.begin();

        for (; vIter != v.end() && proxIter != prox.end() && thresholdsIter != thresholds.end();
             vIter++, thresholdsIter++, proxIter++) {
            if (*vIter > *thresholdsIter) {
                *proxIter = *vIter - *thresholdsIter;
            } else if (*vIter < -*thresholdsIter) {
                *proxIter = *vIter + *thresholdsIter;
            } else {
                *proxIter = 0;
            }
        }
    }

    template <typename data_t>
    auto SoftThresholding<data_t>::cloneImpl() const -> SoftThresholding<data_t>*
    {
        return new SoftThresholding<data_t>(this->getRangeDescriptor());
    }

    template <typename data_t>
    auto SoftThresholding<data_t>::isEqual(const ProximityOperator<data_t>& other) const -> bool
    {
        if (!ProximityOperator<data_t>::isEqual(other)) {
            return false;
        }

        auto otherDerived = downcast_safe<SoftThresholding<data_t>>(&other);

        return static_cast<bool>(otherDerived);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class SoftThresholding<float>;
    template class SoftThresholding<double>;
} // namespace elsa
