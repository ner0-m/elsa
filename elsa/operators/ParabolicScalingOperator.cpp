#include "ParabolicScalingOperator.h"
#include "Timer.h"

namespace elsa
{
    template <typename data_t>
    ParabolicScalingOperator<data_t>::ParabolicScalingOperator(const DataDescriptor& descriptor)
        : LinearOperator<data_t>(descriptor, descriptor)
    {
    }

    template <typename data_t>
    void ParabolicScalingOperator<data_t>::applyImpl(const DataContainer<data_t>& x,
                                                     DataContainer<data_t>& Aax) const
    {
        // TODO add logic
        Timer timeguard("ParabolicScalingOperator", "apply");
        Aax = x;
    }

    template <typename data_t>
    void ParabolicScalingOperator<data_t>::applyAdjointImpl(const DataContainer<data_t>& y,
                                                            DataContainer<data_t>& Aaty) const
    {
        // TODO add logic
        Timer timeguard("ParabolicScalingOperator", "applyAdjoint");
        Aaty = y;
    }

    template <typename data_t>
    ParabolicScalingOperator<data_t>* ParabolicScalingOperator<data_t>::cloneImpl() const
    {
        return new ParabolicScalingOperator(this->getDomainDescriptor());
    }

    template <typename data_t>
    bool ParabolicScalingOperator<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        auto otherParabolicScaling = dynamic_cast<const ParabolicScalingOperator*>(&other);
        return static_cast<bool>(otherParabolicScaling);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class ParabolicScalingOperator<float>;
    template class ParabolicScalingOperator<std::complex<float>>;
    template class ParabolicScalingOperator<double>;
    template class ParabolicScalingOperator<std::complex<double>>;
} // namespace elsa
