#include "HardThresholding.h"

namespace elsa
{
    template <typename data_t>
    HardThresholding<data_t>::HardThresholding(const DataDescriptor& descriptor)
        : ProximityOperator<data_t>(descriptor)
    {
    }

    template <typename data_t>
    void HardThresholding<data_t>::applyImpl(const DataContainer<data_t>& v,
                                             geometry::Threshold<data_t> t,
                                             DataContainer<data_t>& prox) const
    {
        if (v.getSize() != prox.getSize()) {
            throw std::logic_error("HardThresholding: sizes of v and prox must match");
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
    auto HardThresholding<data_t>::cloneImpl() const -> HardThresholding<data_t>*
    {
        return new HardThresholding<data_t>(this->getRangeDescriptor());
    }

    template <typename data_t>
    auto HardThresholding<data_t>::isEqual(const ProximityOperator<data_t>& other) const -> bool
    {
        if (!ProximityOperator<data_t>::isEqual(other)) {
            return false;
        }

        auto otherDerived = dynamic_cast<const HardThresholding<data_t>*>(&other);

        return static_cast<bool>(otherDerived);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class HardThresholding<float>;
    template class HardThresholding<double>;
} // namespace elsa
