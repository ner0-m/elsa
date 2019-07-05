#pragma once

#include "Functional.h"

namespace elsa
{
    /**
     * \brief Class representing the l2 norm functional (squared).
     *
     * \author Matthias Wieczorek - initial code
     * \author Maximilian Hornung - modularization
     * \author Tobias Lasser - modernization
     *
     * \tparam data_t data type for the domain of the residual of the functional, defaulting to real_t
     *
     * The l2 norm (squared) functional evaluates to \f$ 0.5 * \sum_{i=1}^n x_i^2 \f$ for \f$ x=(x_i)_{i=1}^n \f$.
     */
    template <typename data_t = real_t>
    class L2NormPow2 : public Functional<data_t> {
    public:
        /**
         * \brief Constructor for the l2 norm (squared) functional, mapping domain vector to a scalar (without a residual)
         *
         * \param[in] domainDescriptor describing the domain of the functional
         */
        explicit L2NormPow2(const DataDescriptor& domainDescriptor);

        /**
         * \brief Constructor for the l2 norm (squared) functional, using a residual as input to map to a scalar
         *
         * \param[in] residual to be used when evaluating the functional (or its derivatives)
         */
        explicit L2NormPow2(const Residual<data_t>& residual);

        /// default destructor
        ~L2NormPow2() override = default;

    protected:
        /// the evaluation of the l2 norm (squared)
        data_t _evaluate(const DataContainer<data_t>& Rx) override;

        /// the computation of the gradient (in place)
        void _getGradientInPlace(DataContainer<data_t>& Rx) override;

        /// the computation of the Hessian
        LinearOperator<data_t> _getHessian(const DataContainer<data_t>& Rx) override;

        /// implement the polymorphic clone operation
        L2NormPow2<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Functional<data_t>& other) const override;
    };

} // namespace elsa
