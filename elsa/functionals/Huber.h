#pragma once

#include "Functional.h"

namespace elsa
{
    /**
     * \brief Class representing the Huber norm.
     *
     * \author Matthias Wieczorek - initial code
     * \author Maximilian Hornung - modularization
     * \author Tobias Lasser - modernization
     *
     * \tparam data_t data type for the domain of the residual of the functional, defaulting to
     * real_t
     *
     * The Huber norm evaluates to \f$ \sum_{i=1}^n \begin{cases} \frac{1}{2} x_i^2 & \text{for }
     * |x_i| \leq \delta \\ \delta\left(|x_i| - \frac{1}{2}\delta\right) & \text{else} \end{cases}
     * \f$ for \f$ x=(x_i)_{i=1}^n \f$ and a cut-off parameter \f$ \delta \f$.
     *
     * Reference: https://doi.org/10.1214%2Faoms%2F1177703732
     */
    template <typename data_t = real_t>
    class Huber : public Functional<data_t>
    {
    public:
        /**
         * \brief Constructor for the Huber functional, mapping domain vector to scalar (without a
         * residual)
         *
         * \param[in] domainDescriptor describing the domain of the functional
         * \param[in] delta parameter for linear/square cutoff (defaults to 1e-6)
         */
        explicit Huber(const DataDescriptor& domainDescriptor,
                       real_t delta = static_cast<real_t>(1e-6));

        /**
         * \brief Constructor for the Huber functional, using a residual as input to map to a scalar
         *
         * \param[in] residual to be used when evaluating the functional (or its derivative)
         * \param[in] delta parameter for linear/square cutoff (defaults to 1e-6)
         */
        explicit Huber(const Residual<data_t>& residual, real_t delta = static_cast<real_t>(1e-6));

        /// default destructor
        ~Huber() override = default;

    protected:
        /// the evaluation of the Huber norm
        data_t evaluateImpl(const DataContainer<data_t>& Rx) override;

        /// the computation of the gradient (in place)
        void getGradientInPlaceImpl(DataContainer<data_t>& Rx) override;

        /// the computation of the Hessian
        LinearOperator<data_t> getHessianImpl(const DataContainer<data_t>& Rx) override;

        /// implement the polymorphic clone operation
        Huber<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Functional<data_t>& other) const override;

    private:
        /// the cut-off delta
        const real_t _delta;
    };

} // namespace elsa
