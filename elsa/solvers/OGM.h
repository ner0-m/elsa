#pragma once

#include <memory>
#include <optional>

#include "Solver.h"
#include "Problem.h"
#include "DataContainer.h"

namespace elsa
{
    /**
     * @brief Class representing the Optimized Gradient Method.
     *
     * This class implements the Optimized Gradient Method as introduced by Kim and Fessler in 2016.
     * OGM is a first order method to efficiently optimize convex functions with
     * Lipschitz-Continuous gradients. It can be seen as an improvement on Nesterov's Fast Gradient
     * Method.
     *
     * @details
     * # Optimized Gradient method algorithm overview #
     * The algorithm repeats the following update steps for \f$i = 0, \dots, N-1\f$
     * \f{align*}{
     * y_{i+1} &= x_i - \frac{1}{L} f'(x_i) \\
     * \Theta_{i+1} &=
     * \begin{cases}
     *     \frac{1 + \sqrt{1 + 4 \Theta_i^2}}{2} & i \leq N - 2 \\
     *     \frac{1 + \sqrt{1 + 8 \Theta_i^2}}{2} & i \leq N - 1 \\
     * \end{cases} \\
     * x_{i+1} &= y_{i} + \frac{\Theta_i - 1}{\Theta_{i+1}}(y_{i+1} - y_i) +
     *     \frac{\Theta_i}{\Theta_{i+1}}(y_{i+1} - x_i)
     * \f}
     * The inputs are \f$f \in C_{L}^{1, 1}(\mathbb{R}^d)\f$, \f$x_0 \in \mathbb{R}^d\f$,
     * \f$y_0 = x_0\f$, \f$t_0 = 1\f$
     *
     * ## Comparison  to Nesterov's Fast Gradient ##
     * The presented algorithm accelerates FGM by introducing an additional momentum term, which
     * doesn't add a great computational amount.
     *
     * ## OGM References ##
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
    class OGM : public Solver<data_t>
    {
    public:
        /// Scalar alias
        using Scalar = typename Solver<data_t>::Scalar;

        /**
         * @brief Constructor for OGM, accepting an optimization problem and, optionally, a value
         * for epsilon
         *
         * @param[in] problem the problem that is supposed to be solved
         * @param[in] epsilon affects the stopping condition
         */
        OGM(const Problem<data_t>& problem,
            data_t epsilon = std::numeric_limits<data_t>::epsilon());

        /**
         * @brief Constructor for OGM, accepting an optimization problem the inverse of a
         * preconditioner and, optionally, a value for epsilon
         *
         * @param[in] problem the problem that is supposed to be solved
         * @param[in] preconditionerInverse the inverse of the preconditioner
         * @param[in] epsilon affects the stopping condition
         */
        OGM(const Problem<data_t>& problem, const LinearOperator<data_t>& preconditionerInverse,
            data_t epsilon = std::numeric_limits<data_t>::epsilon());

        /// make copy constructor deletion explicit
        OGM(const OGM<data_t>&) = delete;

        /// default destructor
        ~OGM() override = default;

        /**
         * @brief Solve the optimization problem, i.e. apply iterations number of iterations of
         * gradient descent
         *
         * @param[in] iterations number of iterations to execute
         * @param[in] x0 optional initial solution, initial solution set to zero if not present
         *
         * @returns the approximated solution
         */
        DataContainer<data_t>
            solve(index_t iterations,
                  std::optional<DataContainer<data_t>> x0 = std::nullopt) override;

    private:
        /// the differentiable optimizaion problem
        std::unique_ptr<Problem<data_t>> _problem;

        /// variable affecting the stopping condition
        data_t _epsilon;

        /// the inverse of the preconditioner (if supplied)
        std::unique_ptr<LinearOperator<data_t>> _preconditionerInverse{};

        /// implement the polymorphic clone operation
        OGM<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Solver<data_t>& other) const override;
    };
} // namespace elsa
