#pragma once

#include <memory>
#include <optional>

#include "Solver.h"
#include "Functional.h"

namespace elsa
{
    /**
     * @brief Class implementing Nonlinear Conjugate Gradients with customizable line search and
     * beta calculation
     *
     * @author Eddie Groh - initial code
     *
     * This Nonlinear CG can minimize any continuous function f for which the the first and second
     * derivative can be computed or approximated. By this usage of the Gradient and Hessian
     * respectively, it will converge to a local minimum near the starting point.
     *
     * Because CG can only generate n conjugate vectors, if the problem has dimension n, it improves
     * convergence to reset the search direction every n iterations, especially for small n.
     * Restarting means that the search direction is "forgotten" and CG is started again in the
     * direction of the steepest descent
     *
     * Convergence is considered reached when \f$ \| f'(x) \| \leq \epsilon \| f'(x_0)} \| \f$
     * satisfied for some small \f$ \epsilon > 0\f$. Here \f$ x \f$ denotes the solution
     * obtained in the last step, and \f$ x_0 \f$ denotes the initial guess.
     *
     * References:
     * https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
     */
    template <typename data_t = real_t>
    class CGNL : public Solver<data_t>
    {
    public:
        /// Scalar alias
        using Scalar = typename Solver<data_t>::Scalar;

        /**
         * @brief Function Object which performs a single line search iteration
         *
         * @param[in] problem the problem that is supposed to be solved
         * @param[in] xVector the current solution
         * @param[in] dVector direction vector of the current CGNL step
         * @param[in] deltaD dot product of the d vector with itself
         *
         * @return[out] a pair consisting of a boolean indicating if the line search has converged
         * and the new xVector after this line search iteration
         */
        using LineSearchFunction = std::function<DataContainer<data_t>(
            Functional<data_t>& functional, const DataContainer<data_t>& xVector,
            const DataContainer<data_t>& dVector, data_t deltaD, data_t epsilon)>;

        /**
         * @brief Function Object which calculates a beta value based on the direction
         * vector and residual vector
         *
         * @param[in] dVector the vector representing the direction of the current CGNL step
         * @param[in] rVector the residual vector representing the negative gradient
         *
         * @return[out] a pair consisting of the calculated beta and the deltaNew
         */
        using BetaFunction = std::function<std::pair<data_t, data_t>(
            const DataContainer<data_t>& dVector, const DataContainer<data_t>& rVector,
            data_t deltaNew)>;

        /**
         * @brief Constructor for CGNL, accepting an optimization problem and, optionally, a
         * value for epsilon.
         *
         * This will use a constant step length as line search function and Polak-Ribière as
         * beta function
         *
         * @param[in] problem the problem that is supposed to be solved
         * @param[in] epsilon affects the stopping condition
         */
        explicit CGNL(const Functional<data_t>& functional,
                      data_t epsilon = std::numeric_limits<data_t>::epsilon());

        /**
         * @brief Constructor for CGNL, accepting an optimization problem and, optionally, a
         * value for epsilon
         *
         * This will use Newton-Raphson as line search function and Polak-Ribière
         * as beta function
         *
         * @param[in] problem the problem that is supposed to be solved
         * @param[in] epsilon affects the stopping condition
         * @param[in] line_search_iterations number of iterations for each line search
         */
        CGNL(const Functional<data_t>& functional, data_t epsilon, index_t line_search_iterations);

        /**
         * @brief Constructor for CGNL, accepting an optimization problem and, optionally, a
         * value for epsilon
         *
         * @param[in] problem the problem that is supposed to be solved
         * @param[in] epsilon affects the stopping condition
         * @param[in] line_search_iterations maximum number of iterations for the line search
         * @param[in] line_search_function function which will be evaluated each
         * @param[in] beta_function affects the stopping condition
         */
        CGNL(const Functional<data_t>& functional, data_t epsilon, index_t line_search_iterations,
             const LineSearchFunction& line_search_function, const BetaFunction& beta_function);

        /// make copy constructor deletion explicit
        CGNL(const CGNL<data_t>&) = delete;

        /// default destructor
        ~CGNL() override = default;

        /**
         * @brief Solve the optimization problem, i.e. apply iterations number of iterations of
         * CGNL
         *
         * @param[in] iterations number of iterations to execute
         * @param[in] xStart optional initial solution, initial solution set to zero if not present
         *
         * @returns the approximated solution
         */
        DataContainer<data_t>
            solve(index_t iterations,
                  std::optional<DataContainer<data_t>> xStart = std::nullopt) override;

        /// Line search Newton-Raphson
        static const inline LineSearchFunction lineSearchNewtonRaphson =
            [](Functional<data_t>& functional, const DataContainer<data_t>& xVectorInput,
               const DataContainer<data_t>& dVector, data_t deltaD,
               data_t epsilon) -> DataContainer<data_t> {
            auto xVector = xVectorInput;
            for (index_t j = 0; j < 10; j++) {
                // alpha <= -([f'(xVector)]^T * d) / (d^T * f''(xVector) * d)
                auto alpha = static_cast<data_t>(-1.0)
                             * functional.getGradient(xVector).dot(dVector)
                             / dVector.dot(functional.getHessian(xVector).apply(dVector));
                // xVector <= xVector + alpha * d
                xVector = xVector + alpha * dVector;
                // calculate if the line search has converged
                bool converged = alpha * alpha * deltaD < epsilon * epsilon;
                if (converged) {
                    return xVector;
                }
            }
            return xVector;
        };

        /// Line search constant step size
        static const inline LineSearchFunction lineSearchConstantStepSize =
            [](const Functional<data_t>&, const DataContainer<data_t>& xVector,
               const DataContainer<data_t>& dVector, data_t, data_t) -> DataContainer<data_t> {
            return xVector + std::numeric_limits<data_t>::epsilon() * 150.0 * dVector;
        };

        /// beta calculation Polak-Ribière
        static const inline BetaFunction betaPolakRibiere =
            [](const DataContainer<data_t>& dVector, const DataContainer<data_t>& rVector,
               data_t deltaNew) -> std::pair<data_t, data_t> {
            // deltaOld <= deltaNew
            auto deltaOld = deltaNew;
            // deltaMid <= r^T * d
            auto deltaMid = rVector.dot(dVector);
            // deltaNew <= r^T * r
            deltaNew = rVector.dot(rVector);

            // beta <= (deltaNew - deltaMid) / deltaOld
            auto beta = (deltaNew - deltaMid) / deltaOld;
            return {beta, deltaNew};
        };

    private:
        /// the differentiable optimization problem
        std::unique_ptr<Functional<data_t>> functional_;

        /// variable affecting the stopping condition
        data_t epsilon_;

        ///
        index_t line_search_iterations_;

        LineSearchFunction line_search_function_;

        BetaFunction beta_function_;

        /// implement the polymorphic clone operation
        CGNL<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Solver<data_t>& other) const override;
    };
} // namespace elsa
