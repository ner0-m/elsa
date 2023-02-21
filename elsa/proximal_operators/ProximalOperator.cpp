#include "ProximalOperator.h"
#include "Timer.h"
#include "elsaDefines.h"

namespace elsa
{
    template <typename data_t>
    ProximalOperator<data_t>::ProximalOperator(const ProximalOperator& other)
        : ptr_(other.ptr_->clone())
    {
    }

    template <typename data_t>
    ProximalOperator<data_t>& ProximalOperator<data_t>::operator=(const ProximalOperator& other)
    {
        ptr_ = other.ptr_->clone();
        return *this;
    }

    template <typename data_t>
    auto ProximalOperator<data_t>::apply(const DataContainer<data_t>& v, SelfType_t<data_t> t) const
        -> DataContainer<data_t>
    {
        return ptr_->apply(v, t);
    }

    template <typename data_t>
    void ProximalOperator<data_t>::apply(const DataContainer<data_t>& v, SelfType_t<data_t> t,
                                         DataContainer<data_t>& prox) const
    {
        Timer timeguard("ProximalOperator", "apply");
        ptr_->apply(v, t, prox);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class ProximalOperator<float>;
    template class ProximalOperator<double>;
} // namespace elsa
