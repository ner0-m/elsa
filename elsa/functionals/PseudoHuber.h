#pragma once

#include "Functional.h"

namespace elsa
{
    /**
     * \brief Class representing the Pseudohuber norm.
     *
     * \author Matthias Wieczorek - initial code
     * \author Maximilian Hornung - modularization
     * \author Tobias Lasser - modernization, fixes
     *
     * \tparam data_t data type for the domain of the residual of the functional, defaulting to
     * real_t
     *
     * The Pseudohuber norm evaluates to \f$ \sum_{i=1}^n \delta \left( \sqrt{1 + (x_i / \delta)^2}
     * - 1 \right) \f$ for \f$ x=(x_i)_{i=1}^n \f$ and a slope parameter \f$ \delta \f$.
     *
     * Reference: https://doi.org/10.1109%2F83.551699
     */
    template <typename data_t = real_t>
    class PseudoHuber : public Functional<data_t>
    {
    public:
        /**
         * \brief Constructor for the Pseudohuber functional, mapping domain vector to scalar
         * (without a residual)
         *
         * \param[in] domainDescriptor describing the domain of the functional
         * \param[in] delta parameter for linear slope (defaults to 1)
         */
        explicit PseudoHuber(const DataDescriptor& domainDescriptor,
                             real_t delta = static_cast<real_t>(1));

        /**
         * \brief Constructor for the Pseudohuber functional, using a residual as input to map to a
         * scalar
         *
         * \param[in] residual to be used when evaluating the functional (or its derivative)
         * \param[in] delta parameter for linear slope (defaults to 1)
         */
        explicit PseudoHuber(const Residual<data_t>& residual,
                             real_t delta = static_cast<real_t>(1));

        /// default destructor
        ~PseudoHuber() override = default;

    protected:
        /// the evaluation of the Huber norm
        data_t evaluateImpl(const DataContainer<data_t>& Rx) override;

        /// the computation of the gradient (in place)
        void getGradientInPlaceImpl(DataContainer<data_t>& Rx) override;

        /// the computation of the Hessian
        LinearOperator<data_t> getHessianImpl(const DataContainer<data_t>& Rx) override;

        /// implement the polymorphic clone operation
        PseudoHuber<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Functional<data_t>& other) const override;

    private:
        /// the slope delta
        const real_t _delta;
    };

} // namespace elsa
