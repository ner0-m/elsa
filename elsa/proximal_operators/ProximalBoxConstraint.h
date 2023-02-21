#include <optional>

#include "DataContainer.h"
#include "StrongTypes.h"

namespace elsa
{
    /// @brief Proximal operator of the indicator functions for box and
    /// non-negativity sets. The proximal operator is the projection of onto
    /// the given set.
    ///
    /// Given the set \f$P\f$ of elements \f$a \leq x \leq b\f$, the proximal operator
    /// for the indicator function:
    /// \f[
    /// f(x) =
    /// \begin{cases}
    ///     0      & \text{if } a \leq x \leq b \text{ everywhere}, \\
    ///     \infty & \text{else}
    /// \end{cases}
    /// \f]
    /// is given by:
    /// \f[
    /// f(x) =
    /// \begin{cases}
    ///     a      & \text{if } x < a, \\
    ///     x      & \text{if } a \leq x \leq b, \\
    ///     b      & \text{if } x > b
    /// \end{cases}
    /// \f]
    /// The proximal operator is independent for each dimension, and hence,
    /// can be computed coefficient wise.
    ///
    /// Note: the proximal operator is independent of the step length
    ///
    /// @see IndicatorBox IndicatorNonNegativity
    template <class data_t>
    class ProximalBoxConstraint
    {
    public:
        /// defaulted default constructor
        ProximalBoxConstraint() = default;

        /// defaulted copy constructor
        ProximalBoxConstraint(const ProximalBoxConstraint<data_t>&) = default;

        /// defaulted copy assignment operator
        ProximalBoxConstraint& operator=(const ProximalBoxConstraint<data_t>&) = default;

        /// defaulted move constructor
        ProximalBoxConstraint(ProximalBoxConstraint<data_t>&&) noexcept = default;

        /// defaulted move assignment operator
        ProximalBoxConstraint& operator=(ProximalBoxConstraint<data_t>&&) noexcept = default;

        /// defaulted deconstructor
        ~ProximalBoxConstraint() = default;

        /// Construct proximal operator with only a lower bound
        explicit ProximalBoxConstraint(data_t lower);

        /// Construct proximal operator with a lower and upper bound
        ProximalBoxConstraint(data_t lower, data_t upper);

        /// Apply proximal operator to the given vector
        DataContainer<data_t> apply(const DataContainer<data_t>& v, SelfType_t<data_t> t) const;

        /// Apply proximal operator to the given vector, by projection the
        /// vector v onto the given set
        void apply(const DataContainer<data_t>& v, SelfType_t<data_t> t,
                   DataContainer<data_t>& prox) const;

        friend bool operator==(const ProximalBoxConstraint<data_t>&,
                               const ProximalBoxConstraint<data_t>&);

        friend bool operator!=(const ProximalBoxConstraint<data_t>&,
                               const ProximalBoxConstraint<data_t>&);

    private:
        std::optional<data_t> lower_ = {};
        std::optional<data_t> upper_ = {};
    };
} // namespace elsa
