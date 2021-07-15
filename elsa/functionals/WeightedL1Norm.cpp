#include "WeightedL1Norm.h"
#include "LinearOperator.h"

namespace elsa
{
    template <typename data_t>
    WeightedL1Norm<data_t>::WeightedL1Norm(const DataContainer<data_t>& weightingOp)
        : Functional<data_t>(weightingOp.getDataDescriptor()), _weightingOp{weightingOp}
    {
        // sanity check
        for (data_t weight : weightingOp) {
            if (weight < 0) {
                throw InvalidArgumentError(
                    "WeightedL1Norm: all weights in the w vector should be >= 0");
            }
        }
    }

    template <typename data_t>
    WeightedL1Norm<data_t>::WeightedL1Norm(const Residual<data_t>& residual,
                                           const DataContainer<data_t>& weightingOp)
        : Functional<data_t>(residual), _weightingOp{weightingOp}
    {
        // sanity check
        if (residual.getRangeDescriptor().getNumberOfCoefficients()
            != weightingOp.getDataDescriptor().getNumberOfCoefficients())
            throw InvalidArgumentError(
                "WeightedL1Norm: sizes of residual and weighting operator do not match");
        // sanity check
        for (data_t weight : weightingOp) {
            if (weight < 0) {
                throw InvalidArgumentError(
                    "WeightedL1Norm: all weights in the w vector should be >= 0");
            }
        }
    }

    template <typename data_t>
    const DataContainer<data_t>& WeightedL1Norm<data_t>::getWeightingOperator() const
    {
        return _weightingOp;
    }

    template <typename data_t>
    data_t WeightedL1Norm<data_t>::evaluateImpl(const DataContainer<data_t>& Rx)
    {
        // TODO
        return _weightingOp.dot(abs(Rx));
    }

    template <typename data_t>
    void WeightedL1Norm<data_t>::getGradientInPlaceImpl([[maybe_unused]] DataContainer<data_t>& Rx)
    {
        // TODO
        throw LogicError("WeightedL1Norm: not differentiable, so no gradient! (busted!)");
    }

    template <typename data_t>
    LinearOperator<data_t>
        WeightedL1Norm<data_t>::getHessianImpl([[maybe_unused]] const DataContainer<data_t>& Rx)
    {
        // TODO
        throw LogicError("WeightedL1Norm: not differentiable, so no Hessian! (busted!)");
    }

    template <typename data_t>
    WeightedL1Norm<data_t>* WeightedL1Norm<data_t>::cloneImpl() const
    {
        return new WeightedL1Norm(this->getResidual(), _weightingOp);
    }

    template <typename data_t>
    bool WeightedL1Norm<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (!Functional<data_t>::isEqual(other))
            return false;

        auto otherWL1 = dynamic_cast<const WeightedL1Norm*>(&other);
        if (!otherWL1)
            return false;

        if (_weightingOp != otherWL1->_weightingOp)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class WeightedL1Norm<float>;
    template class WeightedL1Norm<double>;
    //    template class WeightedL1Norm<std::complex<float>>;
    //    template class WeightedL1Norm<std::complex<double>>;
} // namespace elsa
