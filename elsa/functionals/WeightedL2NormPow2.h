#pragma once

#include "Functional.h"
#include "Scaling.h"

namespace elsa
{
    /**
     * \brief Class representing a weighted, squared l2 norm functional.
     *
     * \author Matthias Wieczorek - initial code
     * \author Maximilian Hornung - modularization
     * \author Tobias Lasser - modernization
     *
     * \tparam data_t data type for the domain of the functional, defaulting to real_t
     *
     * The weighted, squared l2 norm functional evaluates to \f$ 0.5 * \| x \|_{W,2} = 0.5 * \langle x, Wx \rangle \f$
     * using the standard scalar product, and where W is a diagonal scaling operator.
     */
    template <typename data_t = real_t>
    class WeightedL2NormPow2 : public Functional<data_t> {
    public:
        /**
         * \brief Constructor for the weighted, squared l2 norm, mapping domain vector to a scalar (without a residual)
         *
         * \param[in] weightingOp diagonal scaling operator used for weights
         */
        explicit WeightedL2NormPow2(const Scaling<data_t>& weightingOp);

        /**
         * \brief Constructor for the weighted, squared l2 norm, using a residual as input to map to a scalar
         *
         * \param[in] residual to be used when evaluating the functional (or its derivatives)
         * \param[in] weightingOp diagonal scaling operator used for weights
         */
        WeightedL2NormPow2(const Residual<data_t>& residual, const Scaling<data_t>& weightingOp);

        /// default destructor
        ~WeightedL2NormPow2() override = default;

        /// returns the weighting operator
        const Scaling<data_t>& getWeightingOperator() const;

    protected:
        /// the evaluation of the weighted, squared l2 norm
        data_t _evaluate(const DataContainer<data_t>& Rx) override;

        /// the computation of the gradient (in place)
        void _getGradientInPlace(DataContainer<data_t>& Rx) override;

        /// the computation of the Hessian
        LinearOperator<data_t> _getHessian(const DataContainer<data_t>& Rx) override;

        /// implement the polymorphic clone operation
        WeightedL2NormPow2<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Functional<data_t>& other) const override;

    private:
        /// the weighting operator
        std::unique_ptr<Scaling<data_t>> _weightingOp;
    };

} // namespace elsa
