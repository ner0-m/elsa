#pragma once

#include "Solver.h"

namespace elsa
{
    /**
     * @brief Class representing a simple gradient descent solver with a fixed, given step size.
     *
     * @author Tobias Lasser - initial code
     *
     * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
     *
     * This class implements a simple gradient descent iterative solver with a fixed, given step
     * size. No particular stopping rule is currently implemented (only a fixed number of
     * iterations, default to 100).
     */
    template <typename data_t = real_t>
    class GradientDescent : public Solver<data_t>
    {
    public:
        /// Scalar alias
        using Scalar = typename Solver<data_t>::Scalar;

        /**
         * @brief Constructor for gradient descent, accepting a problem and a fixed step size
         *
         * @param[in] problem the problem that is supposed to be solved
         * @param[in] stepSize the fixed step size to be used while solving
         */
        GradientDescent(const Problem<data_t>& problem, data_t stepSize);

        /**
         * @brief Constructor for gradient descent, accepting a problem. The step size will be
         * computed as \f$ 1 \over L \f$ with \f$ L \f$ being the Lipschitz constant of the
         * function.
         *
         * @param[in] problem the problem that is supposed to be solved
         */
        GradientDescent(const Problem<data_t>& problem);

        /// make copy constructor deletion explicit
        GradientDescent(const GradientDescent<data_t>&) = delete;

        /// default destructor
        ~GradientDescent() override = default;

        /// lift the base class method getCurrentSolution
        using Solver<data_t>::getCurrentSolution;

    private:
        /// the step size
        data_t _stepSize;

        /// the default number of iterations
        const index_t _defaultIterations{100};

        /// lift the base class variable _problem
        using Solver<data_t>::_problem;

        /**
         * @brief Solve the optimization problem, i.e. apply iterations number of iterations of
         * gradient descent
         *
         * @param[in] iterations number of iterations to execute (the default 0 value executes
         * _defaultIterations of iterations)
         *
         * @returns a reference to the current solution
         */
        DataContainer<data_t>& solveImpl(index_t iterations) override;

        /// implement the polymorphic clone operation
        GradientDescent<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Solver<data_t>& other) const override;
    };
} // namespace elsa
