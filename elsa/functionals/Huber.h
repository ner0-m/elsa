#pragma once

#include "DataContainer.h"
#include "Functional.h"

namespace elsa
{
    /**
     * @brief Class representing the Huber loss.
     *
     * The Huber loss evaluates to \f$ \sum_{i=1}^n \begin{cases} \frac{1}{2} x_i^2 & \text{for }
     * |x_i| \leq \delta \\ \delta\left(|x_i| - \frac{1}{2}\delta\right) & \text{else} \end{cases}
     * \f$ for \f$ x=(x_i)_{i=1}^n \f$ and a cut-off parameter \f$ \delta \f$.
     *
     * Reference: https://doi.org/10.1214%2Faoms%2F1177703732
     *
     * @tparam data_t data type for the domain of the residual of the functional, defaulting to
     * real_t
     *
     * @author
     * * Matthias Wieczorek - initial code
     * * Maximilian Hornung - modularization
     * * Tobias Lasser - modernization
     *
     */
    template <typename data_t = real_t>
    class Huber : public Functional<data_t>
    {
    public:
        /**
         * @brief Constructor for the Huber functional, mapping domain vector to scalar (without a
         * residual)
         *
         * @param[in] domainDescriptor describing the domain of the functional
         * @param[in] delta parameter for linear/square cutoff (defaults to 1e-6)
         */
        explicit Huber(const DataDescriptor& domainDescriptor,
                       real_t delta = static_cast<real_t>(1e-6));

        /// make copy constructor deletion explicit
        Huber(const Huber<data_t>&) = delete;

        /// default destructor
        ~Huber() override = default;

    protected:
        /// the evaluation of the Huber loss
        data_t evaluateImpl(const DataContainer<data_t>& Rx) override;

        /// the computation of the gradient (in place)
        void getGradientImpl(const DataContainer<data_t>& Rx, DataContainer<data_t>& out) override;

        /// the computation of the Hessian
        LinearOperator<data_t> getHessianImpl(const DataContainer<data_t>& Rx) override;

        /// implement the polymorphic clone operation
        Huber<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Functional<data_t>& other) const override;

    private:
        /// the cut-off delta
        data_t delta_;
    };

} // namespace elsa
