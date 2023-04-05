#pragma once

#include "Functional.h"
#include "LinearOperator.h"

#include <memory>

namespace elsa
{
    /**
     * @brief Class representing a negative log-likelihood functional for emission tomography.
     *
     * The EmissionLogLikelihood functional evaluates as \f$ \sum_{i=1}^n (x_i + r_i) - y_i\log(x_i
     * + r_i) \f$, with \f$ y=(y_i) \f$ denoting the measurements, \f$ r=(r_i) \f$ denoting the mean
     * number of background events, and \f$ x=(x_i) \f$.
     *
     * Typically, \f$ x \f$ is wrapped in a LinearResidual without a data vector, i.e. \f$ x \mapsto
     * Ax \f$.
     *
     * @tparam data_t data type for the domain of the residual of the functional, defaulting to
     * real_t
     *
     * @author
     * * Matthias Wieczorek - initial code
     * * Maximilian Hornung - modularization
     * * Tobias Lasser - rewrite
     *
     */
    template <typename data_t = real_t>
    class EmissionLogLikelihood : public Functional<data_t>
    {
    public:
        /**
         * @brief Constructor for emission log-likelihood, using only y, and a residual as input
         *
         * @param[in] residual to be used when evaluating the functional (or its derivative)
         * @param[in] y the measurement data vector
         */
        EmissionLogLikelihood(const LinearOperator<data_t>& A, const DataContainer<data_t>& y);

        /**
         * @brief Constructor for emission log-likelihood, using y and r, and a residual as input
         *
         * @param[in] residual to be used when evaluating the functional (or its derivative)
         * @param[in] y the measurement data vector
         * @param[in] r the background event data vector
         */
        EmissionLogLikelihood(const LinearOperator<data_t>& A, const DataContainer<data_t>& y,
                              const DataContainer<data_t>& r);

        /// make copy constructor deletion explicit
        EmissionLogLikelihood(const EmissionLogLikelihood<data_t>&) = delete;

        bool isDifferentiable() const override;

        /// default destructor
        ~EmissionLogLikelihood() override = default;

    protected:
        /// the evaluation of the emission log-likelihood
        data_t evaluateImpl(const DataContainer<data_t>& Rx) override;

        /// the computation of the gradient (in place)
        void getGradientImpl(const DataContainer<data_t>& Rx, DataContainer<data_t>& out) override;

        /// the computation of the Hessian
        LinearOperator<data_t> getHessianImpl(const DataContainer<data_t>& Rx) override;

        /// implement the polymorphic clone operation
        EmissionLogLikelihood<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Functional<data_t>& other) const override;

    private:
        /// optional linear operator to apply to x
        std::unique_ptr<LinearOperator<data_t>> A_{};

        /// the measurement data vector y
        DataContainer<data_t> y_;

        /// the background event data vector r
        std::optional<DataContainer<data_t>> r_{};
    };

} // namespace elsa
