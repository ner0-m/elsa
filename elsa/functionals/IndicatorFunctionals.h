#pragma once

#include "DataContainer.h"
#include "DataDescriptor.h"
#include "Functional.h"
#include <limits>

namespace elsa
{
    /// @brief Indicator function for some box shaped set.
    ///
    /// The indicator function with the lower bound \f$a\f$ and the upper bound
    /// \f$b\f$ is given by:
    /// \f[
    /// f(x) =
    /// \begin{cases}
    ///     0      & \text{if } a \leq x \leq b \text{ everywhere}, \\
    ///     \infty & \text{else}
    /// \end{cases}
    /// \f]
    template <class data_t>
    class IndicatorBox final : public Functional<data_t>
    {
    public:
        /// Construct indicator function with \f$-\infty\f$ and \f$\infty\f$ bounds
        explicit IndicatorBox(const DataDescriptor& desc);

        /// Construct indicator function with given bounds
        IndicatorBox(const DataDescriptor& desc, SelfType_t<data_t> lower,
                     SelfType_t<data_t> upper);

    private:
        /// Evaluate the functional
        data_t evaluateImpl(const DataContainer<data_t>& Rx) override;

        /// The gradient functions throws, the indicator function has no gradient
        void getGradientImpl(const DataContainer<data_t>& Rx, DataContainer<data_t>&) override;

        /// The gradient functions throws, the indicator function has no hessian
        LinearOperator<data_t> getHessianImpl(const DataContainer<data_t>& Rx) override;

        /// Implementation of polymorphic clone
        IndicatorBox<data_t>* cloneImpl() const override;

        /// Implementation of polymorphic equality
        bool isEqual(const Functional<data_t>& other) const override;

        /// Lower bound
        data_t lower_ = -std::numeric_limits<data_t>::infinity();

        /// Upper bound
        data_t upper_ = std::numeric_limits<data_t>::infinity();
    };

    /// @brief Indicator function for the set of non-negative numbers.
    ///
    /// The nonnegativity indicator for the set of non-negative numbers is defined as:
    /// \f[
    /// f(x) =
    /// \begin{cases}
    ///     0      & \text{if } 0 \leq x \text{ everywhere}, \\
    ///     \infty & \text{else}
    /// \end{cases}
    /// \f]
    template <class data_t>
    class IndicatorNonNegativity final : public Functional<data_t>
    {
    public:
        /// Construct non-negativity indicator functional
        explicit IndicatorNonNegativity(const DataDescriptor& desc);

    private:
        /// Evaluate the functional
        data_t evaluateImpl(const DataContainer<data_t>& Rx) override;

        /// The gradient functions throws, the indicator function has no gradient
        void getGradientImpl(const DataContainer<data_t>& Rx, DataContainer<data_t>&) override;

        /// The gradient functions throws, the indicator function has no hessian
        LinearOperator<data_t> getHessianImpl(const DataContainer<data_t>& Rx) override;

        /// Implementation of polymorphic clone
        IndicatorNonNegativity<data_t>* cloneImpl() const override;

        /// Implementation of polymorphic equality
        bool isEqual(const Functional<data_t>& other) const override;
    };
} // namespace elsa
