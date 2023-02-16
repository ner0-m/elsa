#include "WeightedL1Norm.h"
#include "DataContainer.h"
#include "Error.h"
#include "LinearOperator.h"

namespace elsa
{
    template <typename data_t>
    WeightedL1Norm<data_t>::WeightedL1Norm(const DataContainer<data_t>& weightingOp)
        : Functional<data_t>(weightingOp.getDataDescriptor()), _weightingOp{weightingOp}
    {
        // sanity check
        if (weightingOp.minElement() < 0) {
            throw InvalidArgumentError(
                "WeightedL1Norm: all weights in the w vector should be >= 0");
        }
    }

    template <typename data_t>
    const DataContainer<data_t>& WeightedL1Norm<data_t>::getWeightingOperator() const
    {
        return _weightingOp;
    }

    template <typename data_t>
    data_t WeightedL1Norm<data_t>::evaluateImpl(const DataContainer<data_t>& x)
    {
        if (x.getDataDescriptor() != _weightingOp.getDataDescriptor()) {
            throw InvalidArgumentError("WeightedL1Norm: x is not of correct size");
        }

        return _weightingOp.dot(cwiseAbs(x));
    }

    template <typename data_t>
    void WeightedL1Norm<data_t>::getGradientImpl(const DataContainer<data_t>&,
                                                 DataContainer<data_t>&)
    {
        throw LogicError("WeightedL1Norm: not differentiable, so no gradient! (busted!)");
    }

    template <typename data_t>
    LinearOperator<data_t> WeightedL1Norm<data_t>::getHessianImpl(const DataContainer<data_t>&)
    {
        throw LogicError("WeightedL1Norm: not differentiable, so no Hessian! (busted!)");
    }

    template <typename data_t>
    WeightedL1Norm<data_t>* WeightedL1Norm<data_t>::cloneImpl() const
    {
        return new WeightedL1Norm(_weightingOp);
    }

    template <typename data_t>
    bool WeightedL1Norm<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (!Functional<data_t>::isEqual(other))
            return false;

        auto otherWL1 = dynamic_cast<const WeightedL1Norm*>(&other);
        if (!otherWL1)
            return false;

        return _weightingOp == otherWL1->_weightingOp;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class WeightedL1Norm<float>;
    template class WeightedL1Norm<double>;
} // namespace elsa
