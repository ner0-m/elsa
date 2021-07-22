#pragma once
#include "Problem.h"
#include "Solver.h"
#include "LinearOperator.h"
#include "QuadricProblem.h"

namespace elsa
{
    /**
     * @brief Class representing a  non-linear CG solver.
     *
     * @author Maryna Shcherbak - initial code
     *
     * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
     *
     *
     */
    template <typename data_t = real_t>
    class NLCG : public Solver<data_t>
    {
        enum Beta { FR, PR };

    public:
        /**
         * @brief Constructor for NLCG, accepting a problem.
         * @param[in] problem the problem that is supposed to be solved
         * @param[in] version of beta calculation
         */
        NLCG(const Problem<data_t>& problem, Beta beta = Beta::FR);

        NLCG(const Problem<data_t>& problem, const LinearOperator<data_t>& preconditionerInverse);
        /// make copy constructor deletion explicit
        NLCG(const NLCG<data_t>&) = delete;

        /// default destructor
        ~NLCG() override = default;
        /// lift the base class method getCurrentSolution
        using Solver<data_t>::getCurrentSolution;

    private:
        // enum class representing which formula will be used for beta: FR for Fletcher-Reeves, PR
        // for Polak-Ribiere
        Beta _beta;

        /// the inverse of the preconditioner (if supplied)
        std::unique_ptr<LinearOperator<data_t>> _preconditionerInverse{};

        /// the default number of iterations
        const index_t _defaultIterations{100};

        /// lift the base class variable _problem
        using Solver<data_t>::_problem;
        /**
         * @brief Solve the optimization problem, i.e. apply iterations number of iterations Cimmino
         *
         * @param[in] iterations number of iterations to execute (the default 0 value executes
         * _defaultIterations of iterations)
         *
         * @returns a reference to the current solution
         */
        DataContainer<data_t>& solveImpl(index_t iterations) override;

        /// implement the polymorphic clone operation
        NLCG<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Solver<data_t>& other) const override;
    };
} // namespace elsa