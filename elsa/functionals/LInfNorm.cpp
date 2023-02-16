#include "LInfNorm.h"
#include "DataContainer.h"
#include "TypeCasts.hpp"

#include <stdexcept>

namespace elsa
{
    template <typename data_t>
    LInfNorm<data_t>::LInfNorm(const DataDescriptor& domainDescriptor)
        : Functional<data_t>(domainDescriptor)
    {
    }

    template <typename data_t>
    data_t LInfNorm<data_t>::evaluateImpl(const DataContainer<data_t>& Rx)
    {
        return Rx.lInfNorm();
    }

    template <typename data_t>
    void LInfNorm<data_t>::getGradientImpl(const DataContainer<data_t>&, DataContainer<data_t>&)
    {
        throw LogicError("LInfNorm: not differentiable, so no gradient! (busted!)");
    }

    template <typename data_t>
    LinearOperator<data_t> LInfNorm<data_t>::getHessianImpl(const DataContainer<data_t>&)
    {
        throw LogicError("LInfNorm: not differentiable, so no Hessian! (busted!)");
    }

    template <typename data_t>
    LInfNorm<data_t>* LInfNorm<data_t>::cloneImpl() const
    {
        return new LInfNorm(this->getDomainDescriptor());
    }

    template <typename data_t>
    bool LInfNorm<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (!Functional<data_t>::isEqual(other))
            return false;

        return is<LInfNorm>(other);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class LInfNorm<float>;
    template class LInfNorm<double>;
    template class LInfNorm<complex<float>>;
    template class LInfNorm<complex<double>>;

} // namespace elsa
