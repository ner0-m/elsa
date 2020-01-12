#pragma once

#include "QuadricProblem.h"
#include "Solver.h"
#include "LinearOperator.h"

namespace elsa
{
    /**
     * \brief Class implementing the conjugate gradient method
     *
     * \author Matthias Wieczorek - initial code
     * \author David Frank - modularization and modernization
     * \author Nikola Dinev - rewrite, various enhancements
     *
     * CG is an iterative method for minimizing quadric functionals \f$ \frac{1}{2} x^tAx - x^tb \f$
     * with a symmetric positive-definite operator \f$ A \f$. Some common optimization problems,
     * e.g. weighted least squares or Tikhonov-regularized weighted least squares, can be
     * reformulated as quadric problems, and solved using CG, even with non-spd operators through
     * the normal equation. Conversion to the normal equation is performed automatically, where
     * possible.
     *
     * Convergence is considered reached when \f$ \| Ax - b \| \leq \epsilon \|Ax_0 - b\| \f$ is
     * satisfied for some small \f$ \epsilon > 0\f$. Here \f$ x \f$ denotes the solution
     * obtained in the last step, and \f$ x_0 \f$ denotes the initial guess.
     *
     * References:
     * https://doi.org/10.6028%2Fjres.049.044
     * https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
     */
    template <typename data_t = real_t>
    class CG : public Solver<data_t>
    {
    public:
        /**
         * \brief Constructor for CG, accepting an optimization problem and, optionally, a value for
         * epsilon
         *
         * \param[in] problem the problem that is supposed to be solved
         * \param[in] epsilon affects the stopping condition
         *
         * If the problem is not a QuadricProblem, a conversion will be attempted. Throws if
         * conversion fails. See QuadricProblem for details on problems that are convertible to
         * quadric form.
         */
        explicit CG(const Problem<data_t>& problem,
                    data_t epsilon = std::numeric_limits<data_t>::epsilon());

        /**
         * \brief Constructor for preconditioned CG, accepting an optimization problem, the inverse
         * of the preconditioner, and, optionally, a value for epsilon
         *
         * \param[in] problem the problem that is supposed to be solved
         * \param[in] preconditionerInverse the inverse of the preconditioner
         * \param[in] epsilon affects the stopping condition
         *
         * If the problem is not a QuadricProblem, a conversion will be attempted. Throws if
         * conversion fails. See QuadricProblem for details on problems that are convertible to
         * quadric form.
         */
        CG(const Problem<data_t>& problem, const LinearOperator<data_t>& preconditionerInverse,
           data_t epsilon = std::numeric_limits<data_t>::epsilon());

        /// make copy constructor deletion explicit
        CG(const CG<data_t>&) = delete;

        /// default destructor
        ~CG() override = default;

    private:
        /// the default number of iterations
        const index_t _defaultIterations{100};

        /// the inverse of the preconditioner (if supplied)
        std::unique_ptr<LinearOperator<data_t>> _preconditionerInverse{};

        /// variable affecting the stopping condition
        data_t _epsilon;

        /// lift the base class method getCurrentSolution
        using Solver<data_t>::getCurrentSolution;

        /// lift the base class variable _problem
        using Solver<data_t>::_problem;

        /**
         * \brief Solve the optimization problem, i.e. apply iterations number of iterations of CG
         *
         * \param[in] iterations number of iterations to execute (the default 0 value executes
         * _defaultIterations of iterations)
         *
         * \returns a reference to the current solution
         */
        DataContainer<data_t>& solveImpl(index_t iterations) override;

        /// implement the polymorphic clone operation
        CG<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Solver<data_t>& other) const override;
    };
} // namespace elsa
