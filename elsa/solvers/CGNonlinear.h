#pragma once

#include <memory>
#include <optional>

#include "Solver.h"
#include "QuadricProblem.h"
#include "LinearOperator.h"

namespace elsa
{
    /**
     * @brief Class implementing the Preconditioned Nonlinear Conjugate Gradients with Secant and
     * Polak-Ribière
     *
     * @author Eddie Groh - initial code
     *
     * TODO Explanation
     *
     * Convergence is considered reached when TODO
     *
     * References:
     * TODO
     */
    template <typename data_t = real_t>
    class CGNonlinear : public Solver<data_t>
    {
    public:
        /// Scalar alias
        using Scalar = typename Solver<data_t>::Scalar;

        /**
         * @brief Constructor for CGNonlinear, accepting an optimization problem and, optionally, a
         * value for epsilon
         *
         * @param[in] problem the problem that is supposed to be solved
         * @param[in] epsilon affects the stopping condition
         *
         * TODO
         */
        explicit CGNonlinear(const Problem<data_t>& problem,
                             data_t epsilon = std::numeric_limits<data_t>::epsilon());

        /// make copy constructor deletion explicit
        CGNonlinear(const CGNonlinear<data_t>&) = delete;

        /// default destructor
        ~CGNonlinear() override = default;

        /**
         * @brief Solve the optimization problem, i.e. apply iterations number of iterations of
         * CGNonlinear
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
        /// the differentiable optimization problem
        std::unique_ptr<Problem<data_t>> _problem;

        /// variable affecting the stopping condition
        data_t _epsilon;

        /// implement the polymorphic clone operation
        CGNonlinear<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Solver<data_t>& other) const override;
    };
} // namespace elsa
