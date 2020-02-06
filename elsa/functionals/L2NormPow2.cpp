#include "L2NormPow2.h"

#include "LinearOperator.h"
#include "Identity.h"

namespace elsa
{
    template <typename data_t>
    L2NormPow2<data_t>::L2NormPow2(const DataDescriptor& domainDescriptor)
        : Functional<data_t>(domainDescriptor)
    {
    }

    template <typename data_t>
    L2NormPow2<data_t>::L2NormPow2(const Residual<data_t>& residual) : Functional<data_t>(residual)
    {
    }

    template <typename data_t>
    data_t L2NormPow2<data_t>::evaluateImpl(const DataContainer<data_t>& Rx)
    {
        return static_cast<data_t>(0.5) * Rx.squaredL2Norm();
    }

    template <typename data_t>
    void L2NormPow2<data_t>::getGradientInPlaceImpl([[maybe_unused]] DataContainer<data_t>& Rx)
    {
        // gradient is Rx itself (no need for self-assignment)
    }

    template <typename data_t>
    LinearOperator<data_t> L2NormPow2<data_t>::getHessianImpl(const DataContainer<data_t>& Rx)
    {
        return leaf(Identity<data_t>(Rx.getDataDescriptor()));
    }

    template <typename data_t>
    L2NormPow2<data_t>* L2NormPow2<data_t>::cloneImpl() const
    {
        return new L2NormPow2(this->getResidual());
    }

    template <typename data_t>
    bool L2NormPow2<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (!Functional<data_t>::isEqual(other))
            return false;

        auto otherL2NormPow2 = dynamic_cast<const L2NormPow2*>(&other);
        return static_cast<bool>(otherL2NormPow2);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class L2NormPow2<float>;
    template class L2NormPow2<double>;
    template class L2NormPow2<std::complex<float>>;
    template class L2NormPow2<std::complex<double>>;

} // namespace elsa
