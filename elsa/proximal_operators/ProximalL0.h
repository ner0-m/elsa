#pragma once

#include "DataContainer.h"
#include "StrongTypes.h"

namespace elsa
{
    /**
     * @brief Class representing the proximal operator of the l0 pseudo-norm
     *
     * @author Andi Braimllari - initial code
     *
     * @tparam data_t data type for the values of the operator, defaulting to real_t
     *
     * This class represents the hard thresholding operator, expressed by its apply method through
     * the function i.e. @f$ prox(v) = v·1_{\{|v| > t\}}. @f$
     *
     * References:
     * http://sfb649.wiwi.hu-berlin.de/fedc_homepage/xplore/tutorials/xlghtmlnode93.html
     */
    template <typename data_t = real_t>
    class ProximalL0
    {
    public:
        ProximalL0() = default;

        /// default destructor
        ~ProximalL0() = default;

        /**
         * @brief apply the proximal operator of the l0 pseudo-norm to an element in the operator's
         * domain
         *
         * @param[in] v input DataContainer
         * @param[in] t input Threshold
         */
        DataContainer<data_t> apply(const DataContainer<data_t>& v,
                                    geometry::Threshold<data_t> t) const;

        /**
         * @brief apply the proximal operator of the l0 pseudo-norm to an element in the operator's
         * domain
         *
         * @param[in] v input DataContainer
         * @param[in] t input Threshold
         * @param[out] prox output DataContainer
         */
        void apply(const DataContainer<data_t>& v, geometry::Threshold<data_t> t,
                   DataContainer<data_t>& prox) const;

    };

    template <typename data_t>
    bool operator==(const ProximalL0<data_t>&, const ProximalL0<data_t>&) { return true; }
    template <typename data_t>
    bool operator!=(const ProximalL0<data_t>&, const ProximalL0<data_t>&) { return false; }

    template <class data_t>
    using HardThresholding = ProximalL0<data_t>;
} // namespace elsa
