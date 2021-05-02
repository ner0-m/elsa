#include "ShearingOperator.h"
#include "Timer.h"

namespace elsa
{
    template <typename data_t>
    ShearingOperator<data_t>::ShearingOperator(const DataDescriptor& descriptor)
        : LinearOperator<data_t>(descriptor, descriptor)
    {
    }

    template <typename data_t>
    void ShearingOperator<data_t>::applyImpl(const DataContainer<data_t>& x,
                                             DataContainer<data_t>& Px) const
    {
        // TODO add logic
        Timer timeguard("ShearingOperator", "apply");
        Px = x;
    }

    template <typename data_t>
    void ShearingOperator<data_t>::applyAdjointImpl(const DataContainer<data_t>& y,
                                                    DataContainer<data_t>& Pty) const
    {
        // TODO add logic
        Timer timeguard("ShearingOperator", "applyAdjoint");
        Pty = y;
    }

    template <typename data_t>
    ShearingOperator<data_t>* ShearingOperator<data_t>::cloneImpl() const
    {
        return new ShearingOperator(this->getDomainDescriptor());
    }

    template <typename data_t>
    bool ShearingOperator<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        auto otherProjector = dynamic_cast<const ShearingOperator*>(&other);
        return static_cast<bool>(otherProjector);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class ShearingOperator<float>;
    template class ShearingOperator<std::complex<float>>;
    template class ShearingOperator<double>;
    template class ShearingOperator<std::complex<double>>;
} // namespace elsa
