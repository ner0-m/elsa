#pragma once

#include "DataContainer.h"
#include "StrongTypes.h"

namespace elsa
{
    /**
     * @brief Class representing the proximal operator of the l1 norm
     *
     * @author Andi Braimllari - initial code
     *
     * @tparam data_t data type for the values of the operator, defaulting to real_t
     *
     * This class represents the soft thresholding operator, expressed by its apply method through
     * the function i.e. @f$ prox(v) = sign(v)·(|v| - t)_+. @f$
     *
     * References:
     * http://sfb649.wiwi.hu-berlin.de/fedc_homepage/xplore/tutorials/xlghtmlnode93.html
     */
    template <typename data_t = real_t>
    class ProximalL1
    {
    public:
        ProximalL1() = default;

        /// default destructor
        ~ProximalL1() = default;

        /**
         * @brief apply the proximal operator of the l1 norm to an element in the operator's domain
         *
         * @param[in] v input DataContainer
         * @param[in] t input Threshold
         * @param[out] prox output DataContainer
         */
        void apply(const DataContainer<data_t>& v, geometry::Threshold<data_t> t,
                   DataContainer<data_t>& prox) const;

        DataContainer<data_t> apply(const DataContainer<data_t>& v,
                                    geometry::Threshold<data_t> t) const;

    };
    template <typename T>
    bool operator==(const ProximalL1<T>& lhs, const ProximalL1<T>& rhs) { return true; }
    template <typename T>
    bool operator!=(const ProximalL1<T>& lhs, const ProximalL1<T>& rhs) { return false; }

    template <class data_t>
    using SoftThresholding = ProximalL1<data_t>;
} // namespace elsa
