#pragma once

#include "Solver.h"

namespace elsa
{
    /**
     * @brief Class implementing Nesterov's Fast Gradient Method.
     *
     * This class implements Nesterov's Fast Gradient Method. FGM is a first order method to
     * efficiently optimize convex functions with Lipschitz-Continuous gradients.
     *
     * @details
     * # Algorithm overview #
     * The algorithm repeats the following update steps for \f$i = 0, \dots, N-1\f$
     * \f{align*}{
     * y_{i+1} &= x_i - \frac{1}{L} f'(x_i) \\
     * t_{i+1} &= \frac{1 + \sqrt{1 + 4 t^2_i}}{2} \\
     * x_{i+1} &= y_{i} + \frac{t_i - 1}{t_{i+1}}(y_{i+1} - y_i)
     * \f}
     * The inputs are \f$f \in C_{L}^{1, 1}(\mathbb{R}^d)\f$, \f$x_0 \in \mathbb{R}^d\f$,
     * \f$y_0 = x_0\f$, \f$t_0 = 1\f$
     *
     * The presented (and also implemented) version of the algorithm corresponds to _FGM1_ in the
     * referenced paper.
     *
     * ## Convergence ##
     * Compared to the standard gradient descent, which has a convergence rate of
     * \f$\mathcal{O}(\frac{1}{N})\f$, the Nesterov's gradient method boots the convergence rate to
     * \f$\mathcal{O}(\frac{1}{N}^2)\f$
     *
     * In the current implementation, no particular stopping rule is implemented, only a fixed
     * number of iterations is used.
     *
     * ## References ##
     * - Kim, D., Fessler, J.A. _Optimized first-order methods for smooth convex minimization_
     (2016) https://doi.org/10.1007/s10107-015-0949-3
     *
     * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
     *
     * @see \verbatim embed:rst
     For a basic introduction and problem statement of first-order methods, see
     :ref:`here <elsa-first-order-methods-doc>` \endverbatim
     *
     * @author
     * - Michael Loipführer - initial code
     * - David Frank - Detailed Documentation
     */
    template <typename data_t = real_t>
    class FGM : public Solver<data_t>
    {
    public:
        /// Scalar alias
        using Scalar = typename Solver<data_t>::Scalar;

        /**
         * @brief Constructor for FGM, accepting an optimization problem and, optionally, a value
         * for epsilon
         *
         * @param[in] problem the problem that is supposed to be solved
         * @param[in] epsilon affects the stopping condition
         */
        FGM(const Problem<data_t>& problem,
            data_t epsilon = std::numeric_limits<data_t>::epsilon());

        /**
         * @brief Constructor for FGM, accepting an optimization problem, the inverse of a
         * preconditioner and, optionally, a value for epsilon
         *
         * @param[in] problem the problem that is supposed to be solved
         * @param[in] preconditionerInverse the inverse of the preconditioner
         * @param[in] epsilon affects the stopping condition
         */
        FGM(const Problem<data_t>& problem, const LinearOperator<data_t>& preconditionerInverse,
            data_t epsilon = std::numeric_limits<data_t>::epsilon());

        /// make copy constructor deletion explicit
        FGM(const FGM<data_t>&) = delete;

        /// default destructor
        ~FGM() override = default;

        /// lift the base class method getCurrentSolution
        using Solver<data_t>::getCurrentSolution;

    private:
        /// the default number of iterations
        const index_t _defaultIterations{100};

        /// variable affecting the stopping condition
        data_t _epsilon;

        /// the inverse of the preconditioner (if supplied)
        std::unique_ptr<LinearOperator<data_t>> _preconditionerInverse{};

        /// lift the base class variable _problem
        using Solver<data_t>::_problem;

        /**
         * @brief Solve the optimization problem, i.e. apply iterations number of iterations of
         * gradient descent
         *
         * @param[in] iterations number of iterations to execute (the default 0 value executes
         * _defaultIterations of iterations)
         *
         * @returns a reference to the current solution
         */
        DataContainer<data_t>& solveImpl(index_t iterations) override;

        /// implement the polymorphic clone operation
        FGM<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Solver<data_t>& other) const override;
    };
} // namespace elsa
