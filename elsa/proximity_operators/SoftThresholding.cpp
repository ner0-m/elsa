#include "SoftThresholding.h"

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
            throw std::logic_error("SoftThresholding: sizes of v and prox must match");
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

        auto otherDerived = dynamic_cast<const SoftThresholding<data_t>*>(&other);

        return static_cast<bool>(otherDerived);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class SoftThresholding<float>;
    template class SoftThresholding<double>;
} // namespace elsa
