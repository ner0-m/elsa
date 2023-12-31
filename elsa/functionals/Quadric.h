#pragma once

#include "DataContainer.h"
#include "Functional.h"
#include "LinearResidual.h"

namespace elsa
{
    /**
     * @brief Class representing a quadric functional.
     *
     * The Quadric functional evaluates to \f$ \frac{1}{2} x^tAx - x^tb \f$ for a symmetric positive
     * definite operator A and a vector b.
     *
     * Please note: contrary to other functionals, Quadric does not allow wrapping an explicit
     * residual.
     *
     * @tparam data_t data type for the domain of the residual of the functional, defaulting to
     * real_t
     *
     * @author
     * * Matthias Wieczorek - initial code
     * * Maximilian Hornung - modularization
     * * Tobias Lasser - modernization
     * * Nikola Dinev - add functionality
     *
     */
    template <typename data_t = real_t>
    class Quadric : public Functional<data_t>
    {
    public:
        /**
         * @brief Constructor for the Quadric functional, using operator A and vector b (no
         * residual).
         *
         * @param[in] A the operator (has to be symmetric positive definite)
         * @param[in] b the data vector
         */
        Quadric(const LinearOperator<data_t>& A, const DataContainer<data_t>& b);

        /**
         * @brief Constructor for the Quadric functional \f$ \frac{1}{2} x^tAx \f$ (trivial data
         * vector)
         *
         * @param[in] A the operator (has to be symmetric positive definite)
         */
        explicit Quadric(const LinearOperator<data_t>& A);

        /**
         * @brief Constructor for the Quadric functional \f$ \frac{1}{2} x^tx - x^tb \f$ (trivial
         * operator)
         *
         * @param[in] b the data vector
         */
        explicit Quadric(const DataContainer<data_t>& b);

        /**
         * @brief Constructor for the Quadric functional \f$ \frac{1}{2} x^tx \f$ (trivial operator
         * and data vector)
         *
         * @param[in] domainDescriptor the descriptor of x
         */
        explicit Quadric(const DataDescriptor& domainDescriptor);

        /// make copy constructor deletion explicit
        Quadric(const Quadric<data_t>&) = delete;

        /// default destructor
        ~Quadric() override = default;

        /// returns the residual \f$ Ax - b \f$, which also corresponds to the gradient of the
        /// functional
        const LinearResidual<data_t>& getGradientExpression() const;

        bool isDifferentiable() const override;

    protected:
        /// the evaluation of the Quadric functional
        data_t evaluateImpl(const DataContainer<data_t>& Rx) override;

        /// the computation of the gradient (in place)
        void getGradientImpl(const DataContainer<data_t>& Rx, DataContainer<data_t>& out) override;

        /// the computation of the Hessian
        LinearOperator<data_t> getHessianImpl(const DataContainer<data_t>& Rx) override;

        /// implement the polymorphic clone operation
        Quadric<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Functional<data_t>& other) const override;

    private:
        /// storing A,b in a linear residual
        LinearResidual<data_t> linResidual_;
    };

} // namespace elsa
