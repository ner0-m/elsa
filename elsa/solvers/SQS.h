#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "L2NormPow2.h"
#include "LinearOperator.h"
#include "Solver.h"

namespace elsa
{
    /**
     * @brief Class representing an SQS Solver.
     *
     * This class implements an SQS solver with multiple options for momentum acceleration and
     * ordered subsets.
     *
     * No particular stopping rule is currently implemented (only a fixed number of iterations,
     * default to 100).
     *
     * Currently, this is limited to least square problems.
     *
     * References:
     * https://doi.org/10.1109/TMI.2014.2350962
     *
     * @author Michael Loipführer - initial code
     *
     * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
     */
    template <typename data_t = real_t>
    class SQS : public Solver<data_t>
    {
    public:
        /// Scalar alias
        using Scalar = typename Solver<data_t>::Scalar;

        SQS(const L2NormPow2<data_t>& problem,
            std::vector<std::unique_ptr<L2NormPow2<data_t>>>&& subsets,
            bool momentumAcceleration = true,
            data_t epsilon = std::numeric_limits<data_t>::epsilon());

        SQS(const L2NormPow2<data_t>& problem,
            std::vector<std::unique_ptr<L2NormPow2<data_t>>>&& subsets,
            const LinearOperator<data_t>& preconditioner, bool momentumAcceleration = true,
            data_t epsilon = std::numeric_limits<data_t>::epsilon());

        SQS(const L2NormPow2<data_t>& problem, bool momentumAcceleration = true,
            data_t epsilon = std::numeric_limits<data_t>::epsilon());

        SQS(const L2NormPow2<data_t>& problem, const LinearOperator<data_t>& preconditioner,
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
        std::unique_ptr<L2NormPow2<data_t>> fullProblem_;

        std::vector<std::unique_ptr<L2NormPow2<data_t>>> subsets_{};

        /// variable affecting the stopping condition
        data_t epsilon_;

        /// the preconditioner (if supplied)
        std::unique_ptr<LinearOperator<data_t>> preconditioner_{};

        /// whether to enable momentum acceleration
        bool momentumAcceleration_;

        /// whether to operate in subset based mode
        bool subsetMode_;

        /// implement the polymorphic clone operation
        SQS<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Solver<data_t>& other) const override;
    };
} // namespace elsa
