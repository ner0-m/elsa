#pragma once

#include <memory>
#include <optional>

#include "Solver.h"
#include "Problem.h"

namespace elsa
{
    /**
     * @brief Class representing an SQS Solver.
     *
     * @author Michael Loipführer - initial code
     *
     * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
     *
     * This class implements an SQS solver with multiple options for momentum acceleration and
     * ordered subsets.
     *
     * No particular stopping rule is currently implemented (only a fixed number of iterations,
     * default to 100).
     *
     * References:
     * https://doi.org/10.1109/TMI.2014.2350962
     */
    template <typename data_t = real_t>
    class SQS : public Solver<data_t>
    {
    public:
        /// Scalar alias
        using Scalar = typename Solver<data_t>::Scalar;

        /**
         * @brief Constructor for SQS, accepting an optimization problem and, optionally, a value
         * for epsilon. If the problem passed to the constructor is a SubsetProblem SQS will operate
         * in ordered subset mode.
         *
         * @param[in] problem the problem that is supposed to be solved
         * @param[in] momentumAcceleration whether to enable Nesterov's momentum acceleration
         * @param[in] epsilon affects the stopping condition
         */
        SQS(const Problem<data_t>& problem, bool momentumAcceleration = true,
            data_t epsilon = std::numeric_limits<data_t>::epsilon());

        /**
         * @brief Constructor for SQS, accepting an optimization problem, the inverse of the
         * preconditioner and, optionally, a value for epsilon. If the problem passed to the
         * constructor is a SubsetProblem SQS will operate in ordered subset mode.
         *
         * @param[in] problem the problem that is supposed to be solved
         * @param[in] preconditioner a preconditioner for the problem at hand
         * @param[in] momentumAcceleration whether or not to enable momentum acceleration
         * @param[in] epsilon affects the stopping condition
         */
        SQS(const Problem<data_t>& problem, const LinearOperator<data_t>& preconditioner,
            bool momentumAcceleration = true,
            data_t epsilon = std::numeric_limits<data_t>::epsilon());

        /// make copy constructor deletion explicit
        SQS(const SQS<data_t>&) = delete;

        /// default destructor
        ~SQS() override = default;

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
        /// The optimization to solve
        /// TODO: This might be a nice variant?
        std::unique_ptr<Problem<data_t>> _problem;

        /// variable affecting the stopping condition
        data_t _epsilon;

        /// the preconditioner (if supplied)
        std::unique_ptr<LinearOperator<data_t>> _preconditioner{};

        /// whether to enable momentum acceleration
        bool _momentumAcceleration;

        /// whether to operate in subset based mode
        bool _subsetMode{false};

        /// implement the polymorphic clone operation
        SQS<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Solver<data_t>& other) const override;
    };
} // namespace elsa
