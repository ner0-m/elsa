#pragma once

#include "elsaDefines.h"
#include "DataContainer.h"
#include "Cloneable.h"

namespace elsa
{
    /**
     * @brief Base class representing a solver for an optimization problem.
     *
     * @author Matthias Wieczorek - initial code
     * @author Maximilian Hornung - modularization
     * @author Tobias Lasser - rewrite, modernization
     *
     * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
     *
     * This class represents abstract (typically iterative) solvers acting on optimization problems.
     */
    template <typename data_t = real_t>
    class Solver : public Cloneable<Solver<data_t>>
    {
    public:
        /// Scalar alias
        using Scalar = data_t;

        Solver() = default;

        /// default destructor
        ~Solver() override = default;

        /**
         * @brief Solve the optimization problem (most likely iteratively)
         *
         * @param[in] iterations number of iterations to execute (optional argument, the default 0
         * value lets the solve choose how many iterations to execute)
         *
         * @returns a reference to the current solution (after solving)
         */
        virtual DataContainer<data_t> solve(index_t iterations) = 0;
    };
} // namespace elsa
