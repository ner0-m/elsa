#pragma once

#include "DataContainer.h"
#include "Functional.h"
#include "LinearOperator.h"

#include <memory>
#include <optional>

namespace elsa
{
    /**
     * @brief Class representing a negative log-likelihood functional for transmission tomography.
     *
     * @author Matthias Wieczorek - initial code
     * @author Maximilian Hornung - modularization
     * @author Tobias Lasser - rewrite
     *
     * @tparam data_t data type for the domain of the residual of the functional, defaulting to
     * real_t
     *
     * The TransmissionLogLikelihood functional evaluates as \f$ \sum_{i=1}^n (b_i \exp(-x_i) + r_i)
     * - y_i\log(b_i \exp(-x_i) + r_i) \f$, with \f$ b=(b_i) \f$ denoting the mean number of photons
     * per detector (blank scan), \f$ y=(y_i) \f$ denoting the measurements, \f$ r=(r_i) \f$
     * denoting the mean number of background events, and \f$ x=(x_i) \f$.
     *
     * Typically, \f$ x \f$ is wrapped in a LinearResidual without a data vector, i.e. \f$ x \mapsto
     * Ax \f$.
     */
    template <typename data_t = real_t>
    class TransmissionLogLikelihood : public Functional<data_t>
    {
    public:
        /**
         * @brief Constructor for transmission log-likelihood, using only y and b (no residual)
         *
         * @param[in] domainDescriptor describing the domain of the functional
         * @param[in] y the measurement data vector
         * @param[in] b the blank scan data vector
         */
        TransmissionLogLikelihood(const DataDescriptor& domainDescriptor,
                                  const DataContainer<data_t>& y, const DataContainer<data_t>& b);

        /**
         * @brief Constructor for transmission log-likelihood, using y, b, and r (no residual)
         *
         * @param[in] domainDescriptor describing the domain of the functional
         * @param[in] y the measurement data vector
         * @param[in] b the blank scan data vector
         * @param[in] r the background event data vector
         */
        TransmissionLogLikelihood(const DataDescriptor& domainDescriptor,
                                  const DataContainer<data_t>& y, const DataContainer<data_t>& b,
                                  const DataContainer<data_t>& r);

        /**
         * @brief Constructor for transmission log-likelihood, using y and b, and a residual as
         * input
         *
         * @param[in] A linear operator to apply to x
         * @param[in] y the measurement data vector
         * @param[in] b the blank scan data vector
         */
        TransmissionLogLikelihood(const LinearOperator<data_t>& A, const DataContainer<data_t>& y,
                                  const DataContainer<data_t>& b);

        /**
         * @brief Constructor for transmission log-likelihood, using y, b, and r, and a residual as
         * input
         *
         * @param[in] A linear operator to apply to x
         * @param[in] y the measurement data vector
         * @param[in] b the blank scan data vector
         * @param[in] r the background event data vector
         */
        TransmissionLogLikelihood(const LinearOperator<data_t>& A, const DataContainer<data_t>& y,
                                  const DataContainer<data_t>& b, const DataContainer<data_t>& r);

        /// make copy constructor deletion explicit
        TransmissionLogLikelihood(const TransmissionLogLikelihood<data_t>&) = delete;

        /// default destructor
        ~TransmissionLogLikelihood() override = default;

    protected:
        /// the evaluation of the transmission log-likelihood
        data_t evaluateImpl(const DataContainer<data_t>& Rx) override;

        /// the computation of the gradient (in place)
        void getGradientImpl(const DataContainer<data_t>& Rx, DataContainer<data_t>& out) override;

        /// the computation of the Hessian
        LinearOperator<data_t> getHessianImpl(const DataContainer<data_t>& Rx) override;

        /// implement the polymorphic clone operation
        TransmissionLogLikelihood<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Functional<data_t>& other) const override;

    private:
        /// optional linear operator to apply to x
        std::unique_ptr<LinearOperator<data_t>> A_{};

        /// the measurement data vector y
        DataContainer<data_t> y_;

        /// the blank scan data vector b
        DataContainer<data_t> b_;

        /// the background event data vector r
        std::optional<DataContainer<data_t>> r_{};
    };
} // namespace elsa
