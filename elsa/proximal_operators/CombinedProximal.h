#pragma once

#include "ProximalOperator.h"
#include "DataContainer.h"
#include "StrongTypes.h"
#include "elsaDefines.h"

namespace elsa
{
    /**
     * @brief Combine multiple proximals operators into a diagnional operator, or
     * as a block operator.
     *
     * Given the proximal operators for a sequence of \f$k\f$ functions \f$ (g_i)_{i=1}^k\f$, then
     * the combined proximal \f$prox_{G}\f$ is
     * \f[ prox_{G}(x) = diag( prox_{g_1}(x_1), \cdots, prox_{g_k}(x_k))
     * \f]
     * where \f$x = (x_1, \dots, x_k)\f$, i.e. represented as a blocked
     * `DataContainer`
     *
     * The proximal operator of a separable sum of functionals, is the combined
     * proximal.
     *
     * @see SeparableSum
     */
    template <class data_t>
    class CombinedProximal
    {
    public:
        CombinedProximal() = default;

        /// Construct a combined proximal from a single proximal
        explicit CombinedProximal(ProximalOperator<data_t> prox);

        /// Construct a combined proximal from a two proximal
        CombinedProximal(ProximalOperator<data_t> prox1, ProximalOperator<data_t> prox2);

        /// Construct a combined proximal from a three proximal
        CombinedProximal(ProximalOperator<data_t> prox1, ProximalOperator<data_t> prox2,
                         ProximalOperator<data_t> prox3);

        /// Construct a combined proximal from a four proximal
        CombinedProximal(ProximalOperator<data_t> prox1, ProximalOperator<data_t> prox2,
                         ProximalOperator<data_t> prox3, ProximalOperator<data_t> prox4);

        /// Construct a combined proximal from a variadic amount of proximal
        template <class... Args>
        CombinedProximal(ProximalOperator<data_t> prox1, ProximalOperator<data_t> prox2,
                         ProximalOperator<data_t> prox3, ProximalOperator<data_t> prox4,
                         ProximalOperator<data_t> prox5, Args... args)
            : proxs_({prox1, prox2, prox3, prox4, prox5, args...})
        {
        }

        /// Apply proximal operator to the given `DataContainer`. It applies
        /// each proximal operator to the corresponding block of the `DataContainer`.
        /// Throws, if the given `DataContainer` does not have a `BlockDataDescriptor`.
        DataContainer<data_t> apply(const DataContainer<data_t>& v, SelfType_t<data_t> t) const;

        /// Apply the proximal operator the to given out parameter.
        ///
        /// @overload
        void apply(const DataContainer<data_t>& v, SelfType_t<data_t> t,
                   DataContainer<data_t>& prox) const;

        /// Get the `i`-th proximal operator
        ProximalOperator<data_t> getIthProximal(index_t i);

        /// Add an additional proximal operator
        void addProximal(ProximalOperator<data_t> prox);

    private:
        std::vector<ProximalOperator<data_t>> proxs_{};
    };

    // user-defined deduction guides:
    template <class data_t>
    CombinedProximal(ProximalOperator<data_t>) -> CombinedProximal<data_t>;

    template <class data_t>
    CombinedProximal(ProximalOperator<data_t>, ProximalOperator<data_t>)
        -> CombinedProximal<data_t>;

    template <class data_t>
    CombinedProximal(ProximalOperator<data_t>, ProximalOperator<data_t>, ProximalOperator<data_t>)
        -> CombinedProximal<data_t>;

    template <class data_t>
    CombinedProximal(ProximalOperator<data_t>, ProximalOperator<data_t>, ProximalOperator<data_t>,
                     ProximalOperator<data_t>) -> CombinedProximal<data_t>;

} // namespace elsa
