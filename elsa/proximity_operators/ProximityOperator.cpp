#include "ProximityOperator.h"
#include "Timer.h"

namespace elsa
{
    template <typename data_t>
    ProximityOperator<data_t>::ProximityOperator(const ProximityOperator& other)
        : ptr_(other.ptr_->clone())
    {
    }

    template <typename data_t>
    ProximityOperator<data_t>& ProximityOperator<data_t>::operator=(const ProximityOperator& other)
    {
        ptr_ = other.ptr_->clone();
        return *this;
    }

    template <typename data_t>
    auto ProximityOperator<data_t>::apply(const DataContainer<data_t>& v,
                                          geometry::Threshold<data_t> t) const
        -> DataContainer<data_t>
    {
        DataContainer<data_t> prox(v.getDataDescriptor());
        apply(v, t, prox);
        return prox;
    }

    template <typename data_t>
    void ProximityOperator<data_t>::apply(const DataContainer<data_t>& v,
                                          geometry::Threshold<data_t> t,
                                          DataContainer<data_t>& prox) const
    {
        Timer timeguard("ProximityOperator", "apply");
        ptr_->apply(v, t, prox);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class ProximityOperator<float>;
    template class ProximityOperator<double>;
} // namespace elsa
