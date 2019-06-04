#include "Identity.h"
#include "Timer.h"

namespace elsa
{
    template <typename data_t>
    Identity<data_t>::Identity(const DataDescriptor& descriptor)
        : LinearOperator<data_t>(descriptor, descriptor)
    {}

    template <typename data_t>
    void Identity<data_t>::_apply(const DataContainer<data_t>& x, DataContainer<data_t>& Ax)
    {
        Timer timeguard("Identity", "apply");
        Ax = x;
    }

    template <typename data_t>
    void Identity<data_t>::_applyAdjoint(const DataContainer<data_t>& y, DataContainer<data_t>& Aty)
    {
        Timer timeguard("Identity", "applyAdjoint");
        Aty = y;
    }

    template <typename data_t>
    Identity<data_t>* Identity<data_t>::cloneImpl() const
    {
        return new Identity(this->getDomainDescriptor());
    }


    // ------------------------------------------
    // explicit template instantiation
    template class Identity<float>;
    template class Identity<std::complex<float>>;
    template class Identity<double>;
    template class Identity<std::complex<double>>;

} // namespace elsa
