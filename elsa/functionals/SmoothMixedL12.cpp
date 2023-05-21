#include "SmoothMixedL12.h"
#include "FiniteDifferences.h"
#include "BlockDescriptor.h"

namespace elsa
{
    template <typename data_t>
    SmoothMixedL12<data_t>::SmoothMixedL12(const DataDescriptor& domainDescriptor, data_t epsilon)
        : Functional<data_t>(domainDescriptor), epsilon{epsilon}
    {
    }

    template <typename data_t>
    data_t SmoothMixedL12<data_t>::evaluateImpl(const DataContainer<data_t>& Rx)
    {
        ;
        return Rx.l12SmoothMixedNorm(epsilon);
    }

    template <typename data_t>
    void SmoothMixedL12<data_t>::getGradientImpl(const DataContainer<data_t>& Rx,
                                                 DataContainer<data_t>& out)
    {
        const auto blockDesc = downcast_safe<BlockDescriptor>(Rx.getDataDescriptor().clone());
        if (!blockDesc)
            throw LogicError("DataContainer: cannot get block from not-blocked container");

        auto A_mul_1 = DataContainer<data_t>(Rx.getDataDescriptor());
        data_t multiplier(0);

        for (index_t i = 0; i < blockDesc->getNumberOfBlocks(); ++i) {
            auto temp_l1 = Rx.getBlock(i).l1Norm();
            multiplier += 1 / (temp_l1 + epsilon);
            for (auto& el : A_mul_1.getBlock(i)) {
                el = temp_l1;
            }
        }

        auto X_div_A = Rx / (elsa::cwiseAbs(Rx) + epsilon);
        out = multiplier * (A_mul_1 * X_div_A);
    }
    template <typename data_t>
    LinearOperator<data_t> SmoothMixedL12<data_t>::getHessianImpl(const DataContainer<data_t>& Rx)
    {
        throw LogicError("SmoothMixedL12: not yet implemented");
    }
    template <typename data_t>
    SmoothMixedL12<data_t>* SmoothMixedL12<data_t>::cloneImpl() const
    {
        return new SmoothMixedL12(this->getDomainDescriptor(), epsilon);
    }
    template <typename data_t>
    bool SmoothMixedL12<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (!Functional<data_t>::isEqual(other))
            return false;

        return is<SmoothMixedL12>(other);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class SmoothMixedL12<float>;
    template class SmoothMixedL12<double>;

} // namespace elsa