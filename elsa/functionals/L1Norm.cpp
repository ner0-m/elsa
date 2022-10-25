#include "L1Norm.h"
#include "TypeCasts.hpp"

#include <stdexcept>

namespace elsa
{
    template <typename data_t>
    L1Norm<data_t>::L1Norm(const DataDescriptor& domainDescriptor)
        : Functional<data_t>(domainDescriptor)
    {
    }

    template <typename data_t>
    L1Norm<data_t>::L1Norm(const Residual<data_t>& residual) : Functional<data_t>(residual)
    {
    }

    template <typename data_t>
    data_t L1Norm<data_t>::evaluateImpl(const DataContainer<data_t>& Rx)
    {
        return Rx.l1Norm();
    }

    template <typename data_t>
    void L1Norm<data_t>::getGradientInPlaceImpl([[maybe_unused]] DataContainer<data_t>& Rx)
    {
        throw LogicError("L1Norm: not differentiable, so no gradient! (busted!)");
    }

    template <typename data_t>
    std::unique_ptr<LinearOperator<data_t>>
        L1Norm<data_t>::getHessianImpl([[maybe_unused]] const DataContainer<data_t>& Rx)
    {
        throw LogicError("L1Norm: not differentiable, so no Hessian! (busted!)");
    }

    template <typename data_t>
    L1Norm<data_t>* L1Norm<data_t>::cloneImpl() const
    {
        return new L1Norm(this->getResidual());
    }

    template <typename data_t>
    bool L1Norm<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (!Functional<data_t>::isEqual(other))
            return false;

        return is<L1Norm>(other);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class L1Norm<float>;
    template class L1Norm<double>;
    template class L1Norm<complex<float>>;
    template class L1Norm<complex<double>>;

} // namespace elsa
