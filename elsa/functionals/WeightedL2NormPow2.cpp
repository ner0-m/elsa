#include "WeightedL2NormPow2.h"
#include "LinearOperator.h"

#include <stdexcept>

namespace elsa
{
    template <typename data_t>
    WeightedL2NormPow2<data_t>::WeightedL2NormPow2(const Scaling<data_t>& weightingOp)
        : Functional<data_t>(weightingOp.getDomainDescriptor()),
          _weightingOp{static_cast<Scaling<data_t>*>(weightingOp.clone().release())}
    {
    }

    template <typename data_t>
    WeightedL2NormPow2<data_t>::WeightedL2NormPow2(const Residual<data_t>& residual,
                                                   const Scaling<data_t>& weightingOp)
        : Functional<data_t>(residual),
          _weightingOp{static_cast<Scaling<data_t>*>(weightingOp.clone().release())}
    {
        // sanity check
        if (residual.getDomainDescriptor() != weightingOp.getDomainDescriptor()
            || residual.getRangeDescriptor() != weightingOp.getRangeDescriptor())
            throw std::invalid_argument(
                "WeightedL2NormPow2: sizes of residual and weighting operator do not match");
    }

    template <typename data_t>
    const Scaling<data_t>& WeightedL2NormPow2<data_t>::getWeightingOperator() const
    {
        return *_weightingOp;
    }

    template <typename data_t>
    data_t WeightedL2NormPow2<data_t>::evaluateImpl(const DataContainer<data_t>& Rx)
    {
        auto temp = _weightingOp->apply(Rx);

        return static_cast<data_t>(0.5) * Rx.dot(temp);
    }

    template <typename data_t>
    void WeightedL2NormPow2<data_t>::getGradientInPlaceImpl(DataContainer<data_t>& Rx)
    {
        auto temp = _weightingOp->apply(Rx);
        Rx = temp;
    }

    template <typename data_t>
    LinearOperator<data_t>
        WeightedL2NormPow2<data_t>::getHessianImpl(const DataContainer<data_t>& Rx)
    {
        return leaf(*_weightingOp);
    }

    template <typename data_t>
    WeightedL2NormPow2<data_t>* WeightedL2NormPow2<data_t>::cloneImpl() const
    {
        // this ugly cast has to go away at some point..
        auto* scaling = dynamic_cast<const Scaling<data_t>*>(_weightingOp.get());
        return new WeightedL2NormPow2(this->getResidual(), *scaling);
    }

    template <typename data_t>
    bool WeightedL2NormPow2<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (!Functional<data_t>::isEqual(other))
            return false;

        auto otherWL2 = dynamic_cast<const WeightedL2NormPow2*>(&other);
        if (!otherWL2)
            return false;

        if (*_weightingOp != *otherWL2->_weightingOp)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class WeightedL2NormPow2<float>;
    template class WeightedL2NormPow2<double>;
    template class WeightedL2NormPow2<std::complex<float>>;
    template class WeightedL2NormPow2<std::complex<double>>;

} // namespace elsa
