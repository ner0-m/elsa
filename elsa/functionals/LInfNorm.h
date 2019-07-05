#pragma once

#include "Functional.h"

namespace elsa
{
    /**
     * \brief Class representing the maximum norm functional (l infinity).
     *
     * \author Matthias Wieczorek - initial code
     * \author Maximilian Hornung - modularization
     * \author Tobias Lasser - modernization
     *
     * \tparam data_t data type for the domain of the residual of the functional, defaulting to real_t
     *
     * The linf / max norm functional evaluates to \f$ \max_{i=1,\ldots,n} |x_i| \f$ for \f$ x=(x_i)_{i=1}^n \f$.
     * Please note that it is not differentiable, hence getGradient and getHessian will throw exceptions.
     */
    template <typename data_t = real_t>
    class LInfNorm : public Functional<data_t> {
    public:
        /**
         * \brief Constructor for the linf norm functional, mapping domain vector to scalar (without a residual)
         *
         * \param[in] domainDescriptor describing the domain of the functional
         */
        explicit LInfNorm(const DataDescriptor& domainDescriptor);

        /**
         * \brief Constructor for the linf norm functional, using a residual as input to map to a scalar
         *
         * \param[in] residual to be used when evaluating the functional (or its derivative)
         */
        explicit LInfNorm(const Residual<data_t>& residual);

        /// default destructor
        ~LInfNorm() override = default;

    protected:
        /// the evaluation of the linf norm
        data_t _evaluate(const DataContainer<data_t>& Rx) override;

        /// the computation of the gradient (in place)
        void _getGradientInPlace(DataContainer<data_t>& Rx) override;

        /// the computation of the Hessian
        LinearOperator<data_t> _getHessian(const DataContainer<data_t>& Rx) override;

        /// implement the polymorphic clone operation
        LInfNorm<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Functional<data_t>& other) const override;
    };

} // namespace elsa
