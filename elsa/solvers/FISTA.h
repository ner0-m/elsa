#pragma once

#include "Solver.h"
#include "LinearResidual.h"
#include "StrongTypes.h"
#include "LASSOProblem.h"

namespace elsa
{
    /**
     * @brief Class representing a Fast Iterative Shrinkage-Thresholding Algorithm solver
     *
     * This class represents a FISTA solver i.e.
     *
     *  - @f$ x_{k} = shrinkageOperator(y_k - \mu * A^T (Ay_k - b)) @f$
     *  - @f$ t_{k+1} = @frac{1 + \sqrt{1 + 4 * t_{k}^2}}{2} @f$
     *  - @f$ y_{k+1} = x_{k} + (\frac{t_{k} - 1}{t_{k+1}}) * (x_{k} - x_{k - 1}) @f$
     *
     * in which shrinkageOperator is the SoftThresholding operator defined as @f$
     * shrinkageOperator(z_k) = sign(z_k)·(|z_k| - \mu*\lambda)_+ @f$.
     *
     * FISTA has a worst-case complexity result of @f$ O(1/k^2) @f$.
     *
     * @author Andi Braimllari - initial code
     *
     * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
     *
     * References:
     * http://www.cs.cmu.edu/afs/cs/Web/People/airg/readings/2012_02_21_a_fast_iterative_shrinkage-thresholding.pdf
     * https://arxiv.org/pdf/2008.02683.pdf
     */
    template <typename data_t = real_t>
    class FISTA : public Solver<data_t>
    {
    public:
        /**
         * @brief Constructor for FISTA, accepting a problem, a fixed step size and optionally, a
         * value for epsilon
         *
         * @param[in] problem the problem that is supposed to be solved
         * @param[in] mu the fixed step size to be used while solving
         * @param[in] epsilon affects the stopping condition
         *
         * Conversion to a LASSOProblem will be attempted. Throws if conversion fails. See
         * LASSOProblem for further details.
         */
        FISTA(const Problem<data_t>& problem, geometry::Threshold<data_t> mu,
              data_t epsilon = std::numeric_limits<data_t>::epsilon());

        /**
         * @brief Constructor for FISTA, accepting a problem and optionally, a value for
         * epsilon
         *
         * @param[in] problem the problem that is supposed to be solved
         * @param[in] epsilon affects the stopping condition
         *
         * The step size will be computed as @f$ 1 \over L @f$ with @f$ L @f$ being the Lipschitz
         * constant of the WLSProblem.
         *
         * Conversion to a LASSOProblem will be attempted. Throws if conversion fails. See
         * LASSOProblem for further details.
         */
        FISTA(const Problem<data_t>& problem,
              data_t epsilon = std::numeric_limits<data_t>::epsilon());

        /// make copy constructor deletion explicit
        FISTA(const FISTA<data_t>&) = delete;

        /// default destructor
        ~FISTA() override = default;

    protected:
        /// lift the base class method getCurrentSolution
        using Solver<data_t>::getCurrentSolution;

        /// lift the base class variable _problem
        using Solver<data_t>::_problem;

        /**
         * @brief Solve the optimization problem, i.e. apply iterations number of iterations of
         * FISTA
         *
         * @param[in] iterations number of iterations to execute (the default 0 value executes
         * _defaultIterations of iterations)
         *
         * @returns a reference to the current solution
         */
        auto solveImpl(index_t iterations) -> DataContainer<data_t>& override;

        /// implement the polymorphic clone operation
        auto cloneImpl() const -> FISTA<data_t>* override;

        /// implement the polymorphic comparison operation
        auto isEqual(const Solver<data_t>& other) const -> bool override;

    private:
        /// private constructor called by a public constructor without the step size so that
        /// getLipschitzConstant is called by a LASSOProblem and not by a non-converted Problem
        FISTA(const LASSOProblem<data_t>& lassoProb, data_t epsilon);

        /// the default number of iterations
        const index_t _defaultIterations{100};

        /// the step size
        data_t _mu;

        /// variable affecting the stopping condition
        data_t _epsilon;
    };
} // namespace elsa
