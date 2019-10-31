#include "Identity.h"
#include "Timer.h"

namespace elsa
{
    template <typename data_t>
    Identity<data_t>::Identity(const DataDescriptor& descriptor)
        : LinearOperator<data_t>(descriptor, descriptor)
    {
    }

    template <typename data_t>
    void Identity<data_t>::applyImpl(const DataContainer<data_t>& x,
                                     DataContainer<data_t>& Ax) const
    {
        Timer timeguard("Identity", "apply");
        Ax = x;
    }

    template <typename data_t>
    void Identity<data_t>::applyAdjointImpl(const DataContainer<data_t>& y,
                                            DataContainer<data_t>& Aty) const
    {
        Timer timeguard("Identity", "applyAdjoint");
        Aty = y;
    }

    template <typename data_t>
    Identity<data_t>* Identity<data_t>::cloneImpl() const
    {
        return new Identity(this->getDomainDescriptor());
    }

    template <typename data_t>
    bool Identity<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        auto otherIdentity = dynamic_cast<const Identity*>(&other);
        if (!otherIdentity)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class Identity<float>;
    template class Identity<std::complex<float>>;
    template class Identity<double>;
    template class Identity<std::complex<double>>;

} // namespace elsa
