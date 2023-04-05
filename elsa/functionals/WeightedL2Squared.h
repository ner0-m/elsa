#pragma once

#include "DataContainer.h"
#include "Functional.h"
#include "Scaling.h"

namespace elsa
{
    /**
     * @brief Class representing a weighted, squared l2 norm functional.
     *
     * The weighted, squared l2 norm functional evaluates to \f$ 0.5 * \| x \|_{W,2} = 0.5 * \langle
     * x, Wx \rangle \f$ using the standard scalar product, and where W is a diagonal scaling
     * operator.
     *
     * @tparam data_t data type for the domain of the functional, defaulting to real_t
     *
     * @author
     * * Matthias Wieczorek - initial code
     * * Maximilian Hornung - modularization
     * * Tobias Lasser - modernization
     */
    template <typename data_t = real_t>
    class WeightedL2Squared : public Functional<data_t>
    {
    public:
        /**
         * @brief Constructor for the weighted, squared l2 norm, mapping domain vector to a scalar
         * (without a residual)
         *
         * @param[in] weightingOp diagonal scaling operator used for weights
         */
        explicit WeightedL2Squared(const DataContainer<data_t>& weights);

        /// make copy constructor deletion explicit
        WeightedL2Squared(const WeightedL2Squared<data_t>&) = delete;

        /// default destructor
        ~WeightedL2Squared() override = default;

        bool isDifferentiable() const override;

        /// returns the weighting operator
        Scaling<data_t> getWeightingOperator() const;

    protected:
        /// the evaluation of the weighted, squared l2 norm
        data_t evaluateImpl(const DataContainer<data_t>& Rx) override;

        /// the computation of the gradient (in place)
        void getGradientImpl(const DataContainer<data_t>& Rx, DataContainer<data_t>& out) override;

        /// the computation of the Hessian
        LinearOperator<data_t> getHessianImpl(const DataContainer<data_t>& Rx) override;

        /// implement the polymorphic clone operation
        WeightedL2Squared<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Functional<data_t>& other) const override;

    private:
        /// the weighting operator
        DataContainer<data_t> weights_;
    };

} // namespace elsa
