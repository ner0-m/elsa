#pragma once

#include "LinearOperator.h"
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
        using Callback = std::function<void(const DataContainer<data_t>&, index_t, index_t)>;

        IterativeSolver(const LinearOperator<data_t>& A, const DataContainer<data_t>& b)
            : A{A.clone()}, b{b}
        {
        }

        /// default destructor
        ~IterativeSolver() override = default;

        /**
         * @brief Solve the optimization problem iteratively)
         *
         * @param[in] iterations number of iterations to execute
         * @param[in] x0 optional initial solution, initial solution set to zero if not present
         *
         * @returns the current solution (after solving)
         */
        virtual DataContainer<data_t> solve(index_t iterations,
                                            std::optional<DataContainer<data_t>> x0 = std::nullopt,
                                            std::optional<Callback> callback = std::nullopt)
        {
            reset();
            return run(iterations, x0, callback);
        }

    protected:
        virtual void reset() = 0;
        virtual DataContainer<data_t> step(DataContainer<data_t> state) = 0;

        virtual DataContainer<data_t> run(index_t iterations,
                                          std::optional<DataContainer<data_t>> x0 = std::nullopt,
                                          std::optional<Callback> callback = std::nullopt)
        {
            DataContainer<data_t> state{A->getRangeDescriptor()};

            if (x0) {
                state = x0.value();
            } else {
                state = 0;
            }

            for (index_t i = 0; i < iterations; ++i) {
                state = step(state);
                if (callback.has_value()) {
                    callback.value()(state, i, iterations);
                }
            }

            return state;
        }

        std::unique_ptr<LinearOperator<data_t>> A;
        DataContainer<data_t> b;
        std::optional<Callback> callback;
    };

} // namespace elsa
