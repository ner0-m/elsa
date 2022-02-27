#pragma once

#include "ProximityOperator.h"

namespace elsa
{
    /**
     * @brief Class representing the proximity operator of the l0 pseudo-norm
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
    class HardThresholding : public ProximityOperator<data_t>
    {
    public:
        /**
         * @brief Construct a HardThresholding operator from the given DataDescriptor
         *
         * @param[in] descriptor DataDescriptor describing the operator values
         */
        HardThresholding(const DataDescriptor& descriptor);

        /// default destructor
        ~HardThresholding() override = default;

    protected:
        /**
         * @brief apply the proximity operator of the l0 pseudo-norm to an element in the operator's
         * domain
         *
         * @param[in] v the input DataContainer
         * @param[in] t the input Threshold
         * @param[out] prox the output DataContainer
         */
        void applyImpl(const DataContainer<data_t>& v, geometry::Threshold<data_t> t,
                       DataContainer<data_t>& prox) const override;

        /**
         * @brief apply the element-wise proximity operator of the l0 pseudo-norm to elements from
         * the operator's domain
         *
         * @param[in] v the input DataContainer
         * @param[in] thresholds the input vector<Threshold>
         * @param[out] prox the output DataContainer
         */
        void applyImpl(const DataContainer<data_t>& v,
                       std::vector<geometry::Threshold<data_t>> thresholds,
                       DataContainer<data_t>& prox) const override;

        /// implement the polymorphic clone operation
        auto cloneImpl() const -> HardThresholding<data_t>* override;

        /// implement the polymorphic comparison operation
        auto isEqual(const ProximityOperator<data_t>& other) const -> bool override;
    };
} // namespace elsa
