#pragma once

#include "Functional.h"
#include "LinearResidual.h"

namespace elsa
{
    /**
     * \brief Class representing a quadric functional.
     *
     * \author Matthias Wieczorek - initial code
     * \author Maximilian Hornung - modularization
     * \author Tobias Lasser - modernization
     *
     * \tparam data_t data type for the domain of the residual of the functional, defaulting to real_t
     *
     * The Quadric functional evaluates to \f$ \frac{1}{2} x^tAx + x^tb \f$ for a symmetric positive definite
     * operator A and a vector b.
     *
     * Please note: contrary to other functionals, Quadric does not allow wrapping an explicit residual.
     */
    template <typename data_t = real_t>
    class Quadric : public Functional<data_t> {
    public:
        /**
         * \brief Constructor for the Quadric functional, using operator A and vector b (no residual).
         *
         * \param[in] A the operator (has to be symmetric positive definite)
         * \param[in] b the data vector
         */
        Quadric(const LinearOperator<data_t>& A, const DataContainer<data_t>& b);

        /// default destructor
        ~Quadric() override = default;

    protected:
        /// the evaluation of the Quadric functional
        data_t _evaluate(const DataContainer<data_t>& Rx) override;

        /// the computation of the gradient (in place)
        void _getGradientInPlace(DataContainer<data_t>& Rx) override;

        /// the computation of the Hessian
        LinearOperator<data_t> _getHessian(const DataContainer<data_t>& Rx) override;

        /// implement the polymorphic clone operation
        Quadric<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Functional<data_t>& other) const override;

    private:
        /// storing A,b in a linear residual
        LinearResidual<data_t> _linearResidual;
    };

} // namespace elsa
