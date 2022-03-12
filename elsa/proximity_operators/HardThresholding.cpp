#include "HardThresholding.h"
#include "Error.h"
#include "TypeCasts.hpp"

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
        prox = 0;
        std::copy_if(std::begin(v), std::end(v), std::begin(prox),
                     [t](auto x) { return std::abs(x) >= t; });
    }

    template <typename data_t>
    void HardThresholding<data_t>::applyImpl(const DataContainer<data_t>& v,
                                             std::vector<geometry::Threshold<data_t>> thresholds,
                                             DataContainer<data_t>& prox) const
    {
        std::transform(std::begin(v), std::end(v), std::begin(thresholds), std::begin(prox),
                       [](auto x, auto t) {
                           if (std::abs(x) > t) {
                               return x;
                           } else {
                               return data_t(0);
                           }
                       });
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

        auto otherDerived = downcast_safe<HardThresholding<data_t>>(&other);

        return static_cast<bool>(otherDerived);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class HardThresholding<float>;
    template class HardThresholding<double>;
} // namespace elsa
