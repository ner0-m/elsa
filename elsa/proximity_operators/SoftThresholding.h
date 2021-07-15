#pragma once

#include "ProximityOperator.h"

namespace elsa
{
    /**
     * @brief Class representing the proximity operator of the l1 norm
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
    class SoftThresholding : public ProximityOperator<data_t>
    {
    public:
        /**
         * @brief Construct a SoftThresholding operator from the given DataDescriptor
         *
         * @param[in] descriptor DataDescriptor describing the operator values
         */
        SoftThresholding(const DataDescriptor& descriptor);

        /// default destructor
        ~SoftThresholding() override = default;

    protected:
        /**
         * @brief apply the proximity operator of the l1 norm to an element in the operator's domain
         *
         * @param[in] v input DataContainer
         * @param[in] t input Threshold
         * @param[out] prox output DataContainer
         */
        void applyImpl(const DataContainer<data_t>& v, geometry::Threshold<data_t> t,
                       DataContainer<data_t>& prox) const override;

        /**
         * @brief apply the proximity operator of the l1 norm to an element in the operator's domain
         *
         * @param[in] v input DataContainer
         * @param[in] thresholds input vector<Threshold>
         * @param[out] prox output DataContainer
         */
        void applyImpl(const DataContainer<data_t>& v,
                       std::vector<geometry::Threshold<data_t>> thresholds,
                       DataContainer<data_t>& prox) const override;

        /// implement the polymorphic clone operation
        auto cloneImpl() const -> SoftThresholding<data_t>* override;

        /// implement the polymorphic comparison operation
        auto isEqual(const ProximityOperator<data_t>& other) const -> bool override;
    };
} // namespace elsa
