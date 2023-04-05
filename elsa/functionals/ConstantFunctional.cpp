#include "ConstantFunctional.h"
#include "Error.h"
#include "Functional.h"
#include "TypeCasts.hpp"
#include "elsaDefines.h"
#include <unistd.h>
#include <utility>

namespace elsa
{

    template <typename data_t>
    ConstantFunctional<data_t>::ConstantFunctional(const DataDescriptor& descriptor,
                                                   SelfType_t<data_t> constant)
        : Functional<data_t>(descriptor), constant_(constant)
    {
    }

    template <typename data_t>
    bool ConstantFunctional<data_t>::isDifferentiable() const
    {
        return true;
    }

    template <typename data_t>
    data_t ConstantFunctional<data_t>::getConstant() const
    {
        return constant_;
    }

    template <typename data_t>
    data_t ConstantFunctional<data_t>::evaluateImpl(const DataContainer<data_t>&)
    {
        return constant_;
    }

    template <typename data_t>
    void ConstantFunctional<data_t>::getGradientImpl(const DataContainer<data_t>&,
                                                     DataContainer<data_t>& out)
    {
        out = 0;
    }

    template <typename data_t>
    LinearOperator<data_t>
        ConstantFunctional<data_t>::getHessianImpl(const DataContainer<data_t>& Rx)
    {
        throw NotImplementedError("ConstantFunctional: not twice differentiable");
    }

    template <typename data_t>
    ConstantFunctional<data_t>* ConstantFunctional<data_t>::cloneImpl() const
    {
        return new ConstantFunctional<data_t>(this->getDomainDescriptor(), constant_);
    }

    template <typename data_t>
    bool ConstantFunctional<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (!Functional<data_t>::isEqual(other)) {
            return false;
        }

        auto* fn = downcast<ConstantFunctional<data_t>>(&other);
        return static_cast<bool>(fn) && constant_ == fn->constant_;
    }

    // ------------------------------------------
    // Zero Functional
    template <typename data_t>
    ZeroFunctional<data_t>::ZeroFunctional(const DataDescriptor& descriptor)
        : Functional<data_t>(descriptor)
    {
    }

    template <typename data_t>
    bool ZeroFunctional<data_t>::isDifferentiable() const
    {
        return true;
    }

    template <typename data_t>
    data_t ZeroFunctional<data_t>::evaluateImpl(const DataContainer<data_t>& Rx)
    {
        return 0;
    }

    template <typename data_t>
    void ZeroFunctional<data_t>::getGradientImpl(const DataContainer<data_t>&,
                                                 DataContainer<data_t>& out)
    {
        out = 0;
    }

    template <typename data_t>
    LinearOperator<data_t> ZeroFunctional<data_t>::getHessianImpl(const DataContainer<data_t>& Rx)
    {
        throw NotImplementedError("ZeroFunctional: not twice differentiable");
    }

    template <typename data_t>
    ZeroFunctional<data_t>* ZeroFunctional<data_t>::cloneImpl() const
    {
        return new ZeroFunctional<data_t>(this->getDomainDescriptor());
    }

    template <typename data_t>
    bool ZeroFunctional<data_t>::isEqual(const Functional<data_t>& other) const
    {
        return Functional<data_t>::isEqual(other) && is<ZeroFunctional<data_t>>(other);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class ConstantFunctional<float>;
    template class ConstantFunctional<double>;
    template class ConstantFunctional<complex<float>>;
    template class ConstantFunctional<complex<double>>;

    template class ZeroFunctional<float>;
    template class ZeroFunctional<double>;
    template class ZeroFunctional<complex<float>>;
    template class ZeroFunctional<complex<double>>;
} // namespace elsa
