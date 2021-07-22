#pragma once

#include "Solver.h"
#include "LinearOperator.h"
#include "WLSProblem.h"
#include "LinearResidual.h"
namespace elsa
{
    /**
     * @brief Class representing a Landweber solver with a fixed, given step size.
     *
     * @author Maryna Shcherbak - initial code
     *
     * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
     *
     * This class implements a (projected) Landweber solver with a fixed, given step
     * size. No particular stopping rule is currently implemented (only a fixed number of
     * iterations, default to 100).
     *
     * References:
     * https://www.researchgate.net/publication/228574125_Iterative_algorithms_in_Tomography
     */
    template <typename data_t = real_t>
    class Landweber : public Solver<data_t>
    {
    public:
        // inside the Landweber class:
        enum class Projected : bool { YES = true, NO = false };

        /**
         * @brief Constructor for Landweber, accepting a problem and a fixed step size
         *
         * @param[in] problem the problem that is supposed to be solved
         * @param[in] stepSize the fixed step size to be used while solving
         * @param[in] projected specifies if we are using the projected Landweber

         */
        Landweber(const WLSProblem<data_t>& problem, data_t stepSize, Projected projected);

        /**
         * @brief Constructor for Landweber, accepting a problem. The step size will be
         * computed as \f$ 1 \over L \f$ with \f$ L \f$ being the Lipschitz constant of the
         * function.
         *
         * @param[in] problem the problem that is supposed to be solved
         * @param[in] projected specifies if we are using the projected Landweber
         */
        Landweber(const WLSProblem<data_t>& problem, Projected projected);

        Landweber(const WLSProblem<data_t>& problem);

        /// make copy constructor deletion explicit
        Landweber(const Landweber<data_t>&) = delete;

        /// default destructor
        ~Landweber() override = default;
        /// lift the base class method getCurrentSolution
        using Solver<data_t>::getCurrentSolution;

    private:
        /// the step size
        data_t _stepSize;

        /// the default number of iterations
        const index_t _defaultIterations{100};

        /// lift the base class variable _problem
        using Solver<data_t>::_problem;

        bool _projected;
        /**
         * @brief Solve the optimization problem, i.e. apply iterations number of iterations of the
         * Landweber algorithm
         *
         * @param[in] iterations number of iterations to execute (the default 0 value executes
         * _defaultIterations of iterations)
         *
         * @returns a reference to the current solution
         */
        DataContainer<data_t>& solveImpl(index_t iterations) override;

        /// implement the polymorphic clone operation
        Landweber<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Solver<data_t>& other) const override;
    };
} // namespace elsa
