#include "MixedL21.h"
#include "FiniteDifferences.h"
#include "BlockDescriptor.h"

namespace elsa
{
    template <typename data_t>
    MixedL21<data_t>::MixedL21(const DataDescriptor& domainDescriptor)
        : Functional<data_t>(domainDescriptor)
    {
    }

    template <typename data_t>
    data_t MixedL21<data_t>::evaluateImpl(const DataContainer<data_t>& Rx)
    {
        ;
        return Rx.l21MixedNorm();
    }

    template <typename data_t>
    void MixedL21<data_t>::getGradientImpl(const DataContainer<data_t>& Rx,
                                           DataContainer<data_t>& out)
    {
        throw LogicError("MixedL21: not differentiable, so no gradient! (busted!)");
    }
    template <typename data_t>
    LinearOperator<data_t> MixedL21<data_t>::getHessianImpl(const DataContainer<data_t>& Rx)
    {
        throw LogicError("MixedL21: not differentiable, so no hessian! (busted!)");
    }
    template <typename data_t>
    MixedL21<data_t>* MixedL21<data_t>::cloneImpl() const
    {
        return new MixedL21(this->getDomainDescriptor());
    }
    template <typename data_t>
    bool MixedL21<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (!Functional<data_t>::isEqual(other))
            return false;

        return is<MixedL21>(other);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class MixedL21<float>;
    template class MixedL21<double>;
    template class MixedL21<complex<float>>;
    template class MixedL21<complex<double>>;

} // namespace elsa