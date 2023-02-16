#include "IndicatorFunctionals.h"
#include "DataDescriptor.h"
#include "DataContainer.h"
#include "Error.h"
#include "Functional.h"
#include "elsaDefines.h"
#include <limits>

namespace elsa
{
    // ------------------------------------------
    // IndicatorBox
    template <class data_t>
    IndicatorBox<data_t>::IndicatorBox(const DataDescriptor& desc) : Functional<data_t>(desc)
    {
    }

    template <class data_t>
    IndicatorBox<data_t>::IndicatorBox(const DataDescriptor& desc, SelfType_t<data_t> lower,
                                       SelfType_t<data_t> upper)
        : Functional<data_t>(desc), lower_(lower), upper_(upper)
    {
    }

    template <class data_t>
    data_t IndicatorBox<data_t>::evaluateImpl(const DataContainer<data_t>& Rx)
    {
        // Project the input value onto the box set
        auto projected = ::elsa::clip(Rx, lower_, upper_);

        // Check if anything changed, by computing the distance
        return (projected - Rx).l2Norm() > 0 ? std::numeric_limits<data_t>::infinity() : 0;
    }

    template <class data_t>
    void IndicatorBox<data_t>::getGradientImpl(const DataContainer<data_t>&, DataContainer<data_t>&)
    {
        throw NotImplementedError("IndicatorBox: Not differentiable");
    }

    template <class data_t>
    LinearOperator<data_t> IndicatorBox<data_t>::getHessianImpl(const DataContainer<data_t>& Rx)
    {
        throw NotImplementedError("IndicatorBox: Not differentiable");
    }

    template <class data_t>
    IndicatorBox<data_t>* IndicatorBox<data_t>::cloneImpl() const
    {
        return new IndicatorBox<data_t>(this->getDomainDescriptor(), lower_, upper_);
    }

    template <class data_t>
    bool IndicatorBox<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (!Functional<data_t>::isEqual(other)) {
            return false;
        }

        auto* fn = downcast<IndicatorBox<data_t>>(&other);
        return static_cast<bool>(fn) && lower_ == fn->lower_ && upper_ == fn->upper_;
    }

    // ------------------------------------------
    // IndicatorNonNegativity
    template <class data_t>
    IndicatorNonNegativity<data_t>::IndicatorNonNegativity(const DataDescriptor& desc)
        : Functional<data_t>(desc)
    {
    }

    template <class data_t>
    data_t IndicatorNonNegativity<data_t>::evaluateImpl(const DataContainer<data_t>& Rx)
    {
        constexpr auto infinity = std::numeric_limits<data_t>::infinity();

        // Project the input value onto the box set
        auto projected = ::elsa::clip(Rx, data_t{0}, infinity);

        // Check if anything changed, by computing the distance
        return (projected - Rx).l2Norm() > 0 ? infinity : 0;
    }

    template <class data_t>
    void IndicatorNonNegativity<data_t>::getGradientImpl(const DataContainer<data_t>&,
                                                         DataContainer<data_t>&)
    {
        throw NotImplementedError("IndicatorNonNegativity: Not differentiable");
    }

    template <class data_t>
    LinearOperator<data_t>
        IndicatorNonNegativity<data_t>::getHessianImpl(const DataContainer<data_t>& Rx)
    {
        throw NotImplementedError("IndicatorNonNegativity: Not differentiable");
    }

    template <class data_t>
    IndicatorNonNegativity<data_t>* IndicatorNonNegativity<data_t>::cloneImpl() const
    {
        return new IndicatorNonNegativity<data_t>(this->getDomainDescriptor());
    }

    template <class data_t>
    bool IndicatorNonNegativity<data_t>::isEqual(const Functional<data_t>& other) const
    {
        return Functional<data_t>::isEqual(other) && is<IndicatorNonNegativity<data_t>>(&other);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class IndicatorBox<float>;
    template class IndicatorBox<double>;

    template class IndicatorNonNegativity<float>;
    template class IndicatorNonNegativity<double>;
} // namespace elsa
