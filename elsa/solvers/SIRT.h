#pragma once
#include "WLSProblem.h"
#include "Solver.h"
#include "LinearOperator.h"

namespace elsa
{
    template <typename data_t = real_t>
    class SIRT : public Solver<data_t>
    {
    public:
        /**
         * \brief Constructor for SIRT, accepting a problem.
         * \param[in] problem the problem that is supposed to be solved
         */
        SIRT(const WLSProblem<data_t>& problem);

        /// make copy constructor deletion explicit
        SIRT(const SIRT<data_t>&) = delete;

        /// default destructor
        ~SIRT() override = default;

    private:
        /// the default number of iterations
        const index_t _defaultIterations{100};

        /// lift the base class method getCurrentSolution
        using Solver<data_t>::getCurrentSolution;

        /// lift the base class variable _problem
        using Solver<data_t>::_problem;
        /**
         * \brief Solve the optimization problem, i.e. apply iterations number of iterations of
         * gradient descent
         *
         * \param[in] iterations number of iterations to execute (the default 0 value executes
         * _defaultIterations of iterations)
         *
         * \returns a reference to the current solution
         */
        DataContainer<data_t>& solveImpl(index_t iterations) override;

        /// implement the polymorphic clone operation
        auto cloneImpl() const -> SIRT<data_t>* override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Solver<data_t>& other) const override;
    };

}
