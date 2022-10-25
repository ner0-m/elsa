#pragma once

#include "Functional.h"

namespace elsa
{
    /**
     * @brief Class representing a weighted l1 norm functional.
     *
     * @author Andi Braimllari - initial code
     *
     * @tparam data_t data type for the domain of the functional, defaulting to real_t
     *
     * The weighted l1 norm functional evaluates to @f$ \| x \|_{w,1} = \sum_{i=1}^n w_{i} *
     * |x_{i}| @f$ where @f$ w_{i} >= 0 @f$.
     */
    template <typename data_t = real_t>
    class WeightedL1Norm : public Functional<data_t>
    {
    public:
        /**
         * @brief Constructor for the weighted l1 norm, mapping domain vector to a scalar
         * (without a residual)
         *
         * @param[in] weightingOp container of the weights
         */
        explicit WeightedL1Norm(const DataContainer<data_t>& weightingOp);

        /**
         * @brief Constructor for the weighted l1 norm, using a residual as input to map to
         * a scalar
         *
         * @param[in] residual to be used when evaluating the functional (or its derivatives)
         * @param[in] weightingOp container of the weights
         */
        WeightedL1Norm(const Residual<data_t>& residual, const DataContainer<data_t>& weightingOp);

        /// make copy constructor deletion explicit
        WeightedL1Norm(const WeightedL1Norm<data_t>&) = delete;

        /// default destructor
        ~WeightedL1Norm() override = default;

        /// returns the weighting operator
        const DataContainer<data_t>& getWeightingOperator() const;

    protected:
        /// the evaluation of the weighted l1 norm
        data_t evaluateImpl(const DataContainer<data_t>& Rx) override;

        /// the computation of the gradient (in place)
        void getGradientInPlaceImpl(DataContainer<data_t>& Rx) override;

        /// the computation of the Hessian
        std::unique_ptr<LinearOperator<data_t>>
            getHessianImpl(const DataContainer<data_t>& Rx) override;

        /// implement the polymorphic clone operation
        WeightedL1Norm<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Functional<data_t>& other) const override;

    private:
        /// the weighting operator
        DataContainer<data_t> _weightingOp;
    };
} // namespace elsa
