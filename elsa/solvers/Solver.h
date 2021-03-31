#pragma once

#include "elsaDefines.h"
#include "Cloneable.h"
#include "Problem.h"

namespace elsa
{
    /**
     * \brief Base class representing a solver for an optimization problem.
     *
     * \author Matthias Wieczorek - initial code
     * \author Maximilian Hornung - modularization
     * \author Tobias Lasser - rewrite, modernization
     *
     * \tparam data_t data type for the domain and range of the problem, defaulting to real_t
     *
     * This class represents abstract (typically iterative) solvers acting on optimization problems.
     */
    template <typename data_t = real_t>
    class Solver : public Cloneable<Solver<data_t>>
    {
    public:
        /**
         * \brief Constructor for the solver, accepting an optimization problem
         *
         * \param[in] problem the problem that is supposed to be solved
         */
        explicit Solver(const Problem<data_t>& problem);

        /// default destructor
        ~Solver() override = default;

        /// return the current estimated solution (const version)
        const DataContainer<data_t>& getCurrentSolution() const;

        /// return the current estimated solution
        DataContainer<data_t>& getCurrentSolution();

        /**
         * \brief Solve the optimization problem (most likely iteratively)
         *
         * \param[in] iterations number of iterations to execute (optional argument, the default 0
         * value lets the solve choose how many iterations to execute)
         *
         * \returns a reference to the current solution (after solving)
         *
         * Please note: this method calls solveImpl, which has to be overridden in derived classes.
         */
        DataContainer<data_t>& solve(index_t iterations = 0);

    protected:
        /// the optimization problem
        std::unique_ptr<Problem<data_t>> _problem;

        /// the solve method to be overridden in derived classes
        virtual DataContainer<data_t>& solveImpl(index_t iterations) = 0;

        /// implement the polymorphic comparison operation
        bool isEqual(const Solver<data_t>& other) const override;
    };
} // namespace elsa
