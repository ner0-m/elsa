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
        std::transform(std::begin(v), std::end(v), std::begin(prox),
                       [t](auto x) {
                           if (x > t) {
                               return x - t;
                           } else if (t < -x) {
                               return x + t;
                           } else {
                               return data_t(0);
                           }
                       });
    }

    template <typename data_t>
    void SoftThresholding<data_t>::applyImpl(const DataContainer<data_t>& v,
                                             std::vector<geometry::Threshold<data_t>> thresholds,
                                             DataContainer<data_t>& prox) const
    {
        std::transform(std::begin(v), std::end(v), std::begin(thresholds), std::begin(prox),
                       [](auto x, auto t) {
                           if (x > t) {
                               return x - t;
                           } else if (x < -t) {
                               return x + t;
                           } else {
                               return data_t(0);
                           }
                       });
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
