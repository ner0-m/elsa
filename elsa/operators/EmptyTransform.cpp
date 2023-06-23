#include "EmptyTransform.h"
#include "Timer.h"
#include "TypeCasts.hpp"

namespace elsa
{
    template <typename data_t>
    EmptyTransform<data_t>::EmptyTransform(const DataDescriptor& domainDescriptor, const DataDescriptor& rangeDescriptor)
        : LinearOperator<data_t>(domainDescriptor, rangeDescriptor)
    {
    }

    template <typename data_t>
    void EmptyTransform<data_t>::applyImpl(const DataContainer<data_t>& x,
                                     DataContainer<data_t>& Ax) const
    {
        Timer timeguard("Identity", "apply");
        Ax = DataContainer<data_t>(this->getRangeDescriptor());
    }

    template <typename data_t>
    void EmptyTransform<data_t>::applyAdjointImpl(const DataContainer<data_t>& y,
                                            DataContainer<data_t>& Aty) const
    {
        Timer timeguard("Identity", "applyAdjoint");
        //std::cout<<"!!!empty adjoint"<<std::endl;
        Aty = DataContainer<data_t>(this->getDomainDescriptor());
    }

    template <typename data_t>
    EmptyTransform<data_t>* EmptyTransform<data_t>::cloneImpl() const
    {
        return new EmptyTransform(this->getDomainDescriptor(), this->getRangeDescriptor());
    }

    template <typename data_t>
    bool EmptyTransform<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        return is<EmptyTransform>(other);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class EmptyTransform<float>;
    template class EmptyTransform<complex<float>>;
    template class EmptyTransform<double>;
    template class EmptyTransform<complex<double>>;

} // namespace elsa
