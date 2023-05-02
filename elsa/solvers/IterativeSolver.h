#pragma once

#include "elsaDefines.h"
#include "DataContainer.h"
#include "Cloneable.h"
#include <optional>

namespace elsa
{
    /**
     * @brief Base class representing an iterative solver for an optimization problem.
     * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
     *
     * This class represents abstract (typically iterative) solvers acting on optimization problems.
     */
    template <typename data_t = real_t>
    class IterativeSolver : public Cloneable<IterativeSolver<data_t>>
    {
    public:
        /// Scalar alias
        using Scalar = data_t;

        IterativeSolver() = default;

        /// default destructor
        ~IterativeSolver() override = default;

        virtual DataContainer<data_t> step(DataContainer<data_t> state) = 0;

        /**
         * @brief Solve the optimization problem iteratively)
         *
         * @param[in] iterations number of iterations to execute
         * @param[in] x0 optional initial solution, initial solution set to zero if not present
         *
         * @returns the current solution (after solving)
         */
        virtual DataContainer<data_t>
            run(index_t iterations, std::optional<DataContainer<data_t>> x0 = std::nullopt,
                std::optional<std::function<void(const DataContainer<data_t>&, index_t, index_t)>>
                    afterStep = std::nullopt) = 0;

        // TODO: Stopping criterion? solve function?
    };

    template <typename data_t>
    DataContainer<data_t> IterativeSolver<data_t>::run(
        index_t iterations, std::optional<DataContainer<data_t>> x0,
        std::optional<std::function<void(const DataContainer<data_t>&, index_t, index_t)>>
            afterStep)
    {
        auto state = x0.value_or(0);

        for (index_t i = 0; i < iterations; ++i) {
            state = step(state);
            afterStep(state, i, iterations);
        }

        return state;
    }

} // namespace elsa
