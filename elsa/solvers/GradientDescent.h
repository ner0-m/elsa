#pragma once

#include "Solver.h"

namespace elsa
{
    /**
     * \brief Class representing a simple gradient descent solver with a fixed, given step size.
     *
     * \author Tobias Lasser - initial code
     *
     * \tparam data_t data type for the domain and range of the problem, defaulting to real_t
     *
     * This class implements a simple gradient descent iterative solver with a fixed, given step
     * size. No particular stopping rule is currently implemented (only a fixed number of
     * iterations, default to 100).
     */
    template <typename data_t = real_t>
    class GradientDescent : public Solver<data_t>
    {
    public:
        /**
         * \brief Constructor for gradient descent, accepting a problem and a fixed step size
         *
         * \param[in] problem the problem that is supposed to be solved
         * \param[in] stepSize the fixed step size to be used while solving
         */
        GradientDescent(const Problem<data_t>& problem, real_t stepSize);

        /// default destructor
        ~GradientDescent() override = default;

    private:
        /// the step size
        real_t _stepSize;

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
         * \param[in]trackOutput a callback function, it should track the current state of solve, it is executed at the end of each iteration
         * with the current number of iterations and the current reconstruction as input
         * If it returns true the solve is immediatly canceled and the current solution is returned [default nullptr]
         *
         * \returns a reference to the current solution
         */
        DataContainer<data_t>& solveImpl(index_t iterations, std::function<bool(int, DataContainer<data_t>& )> trackOutput) override;

        /// implement the polymorphic clone operation
        GradientDescent<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Solver<data_t>& other) const override;
    };
} // namespace elsa
