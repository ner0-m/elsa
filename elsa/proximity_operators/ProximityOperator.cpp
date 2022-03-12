#include "ProximityOperator.h"
#include "Timer.h"

namespace elsa
{
    template <typename data_t>
    ProximityOperator<data_t>::ProximityOperator(const DataDescriptor& descriptor)
        : _rangeDescriptor{descriptor.clone()}
    {
    }

    template <typename data_t>
    auto ProximityOperator<data_t>::getRangeDescriptor() const -> const DataDescriptor&
    {
        return *_rangeDescriptor;
    }

    template <typename data_t>
    auto ProximityOperator<data_t>::apply(const DataContainer<data_t>& v,
                                          geometry::Threshold<data_t> t) const
        -> DataContainer<data_t>
    {
        Timer timeguard("ProximityOperator", "apply");
        DataContainer<data_t> prox(*_rangeDescriptor, v.getDataHandlerType());
        applyImpl(v, t, prox);
        return prox;
    }

    template <typename data_t>
    auto ProximityOperator<data_t>::apply(const DataContainer<data_t>& v,
                                          std::vector<geometry::Threshold<data_t>> thresholds) const
        -> DataContainer<data_t>
    {
        Timer timeguard("ProximityOperator", "apply");
        DataContainer<data_t> prox(*_rangeDescriptor, v.getDataHandlerType());
        applyImpl(v, thresholds, prox);
        return prox;
    }

    template <typename data_t>
    void ProximityOperator<data_t>::apply(const DataContainer<data_t>& v,
                                          geometry::Threshold<data_t> t,
                                          DataContainer<data_t>& prox) const
    {
        Timer timeguard("ProximityOperator", "apply");

        if (v.getSize() != prox.getSize()) {
            throw LogicError("ProximityOperator: sizes of v and prox must match");
        }

        applyImpl(v, t, prox);
    }

    template <typename data_t>
    void ProximityOperator<data_t>::apply(const DataContainer<data_t>& v,
                                          std::vector<geometry::Threshold<data_t>> thresholds,
                                          DataContainer<data_t>& prox) const
    {
        Timer timeguard("ProximityOperator", "apply");

        if (v.getSize() != prox.getSize()) {
            throw LogicError("ProximityOperator: sizes of v and prox must match");
        }

        if (v.getSize() != thresholds.size()) {
            throw LogicError("ProximityOperator: sizes of v and thresholds must match");
        }

        applyImpl(v, thresholds, prox);
    }

    template <typename data_t>
    auto ProximityOperator<data_t>::isEqual(const ProximityOperator<data_t>& other) const -> bool
    {
        return static_cast<bool>(*_rangeDescriptor == *other._rangeDescriptor);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class ProximityOperator<float>;
    template class ProximityOperator<double>;
} // namespace elsa
