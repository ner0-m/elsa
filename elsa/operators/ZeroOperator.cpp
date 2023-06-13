#include "ZeroOperator.h"
#include "TypeCasts.hpp"

namespace elsa
{
    template <typename data_t>
    ZeroOperator<data_t>::ZeroOperator(const DataDescriptor& domainDescriptor,
                                       const DataDescriptor& rangeDescriptor)
        : LinearOperator<data_t>(domainDescriptor, rangeDescriptor)
    {
    }

    template <typename data_t>
    void ZeroOperator<data_t>::applyImpl([[maybe_unused]] const DataContainer<data_t>& x,
                                         DataContainer<data_t>& Ax) const
    {
        Ax = 0;
    }

    template <typename data_t>
    void ZeroOperator<data_t>::applyAdjointImpl([[maybe_unused]] const DataContainer<data_t>& y,
                                                DataContainer<data_t>& Aty) const
    {
        Aty = 0;
    }

    template <typename data_t>
    ZeroOperator<data_t>* ZeroOperator<data_t>::cloneImpl() const
    {
        return new ZeroOperator(this->getDomainDescriptor(), this->getRangeDescriptor());
    }

    template <typename data_t>
    bool ZeroOperator<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        return is<ZeroOperator>(other);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class ZeroOperator<float>;
    template class ZeroOperator<complex<float>>;
    template class ZeroOperator<double>;
    template class ZeroOperator<complex<double>>;

} // namespace elsa
