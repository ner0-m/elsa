#pragma once

#include <optional>

#include "MaybeUninitialized.hpp"
#include "Solver.h"
#include "StrongTypes.h"
#include "LASSOProblem.h"

namespace elsa
{
    /**
     * @brief Class representing an Iterative Shrinkage-Thresholding Algorithm solver
     *
     * This class represents an ISTA solver i.e.
     *
     *  - @f$ x_{k+1} = shrinkageOperator(x_k - \mu * A^T (Ax_k - b)) @f$
     *
     * in which shrinkageOperator is the SoftThresholding operator defined as @f$
     * shrinkageOperator(z_{k}) = sign(z_{k})·(|z_{k}| - \mu*\lambda)_+ @f$. Each iteration of ISTA
     * involves a gradient descent update followed by a shrinkage/soft-threshold step:
     *
     * ISTA has a worst-case complexity result of @f$ O(1/k) @f$.
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
    class ISTA : public Solver<data_t>
    {
    public:
        /// Scalar alias
        using Scalar = typename Solver<data_t>::Scalar;

        /**
         * @brief Constructor for ISTA, accepting a LASSO problem, a fixed step size and optionally,
         * a value for epsilon
         *
         * @param[in] problem the LASSO problem that is supposed to be solved
         * @param[in] mu the fixed step size to be used while solving
         * @param[in] epsilon affects the stopping condition
         */
        ISTA(const LASSOProblem<data_t>& problem, geometry::Threshold<data_t> mu,
             data_t epsilon = std::numeric_limits<data_t>::epsilon());

        /**
         * @brief Constructor for ISTA, accepting a problem, a fixed step size and optionally, a
         * value for epsilon
         *
         * @param[in] problem the problem that is supposed to be solved
         * @param[in] mu the fixed step size to be used while solving
         * @param[in] epsilon affects the stopping condition
         *
         * Conversion to a LASSOProblem will be attempted. Throws if conversion fails. See
         * LASSOProblem for further details.
         */
        ISTA(const Problem<data_t>& problem, geometry::Threshold<data_t> mu,
             data_t epsilon = std::numeric_limits<data_t>::epsilon());

        /**
         * @brief Constructor for ISTA, accepting a problem and optionally, a value for
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
        ISTA(const Problem<data_t>& problem,
             data_t epsilon = std::numeric_limits<data_t>::epsilon());

        /// make copy constructor deletion explicit
        ISTA(const ISTA<data_t>&) = delete;

        /// default destructor
        ~ISTA() override = default;

        /**
         * @brief Solve the optimization problem, i.e. apply iterations number of iterations of
         * ISTA
         *
         * @param[in] iterations number of iterations to execute
         * @param[in] x0 optional initial solution, initial solution set to zero if not present
         *
         * @returns the approximated solution
         */
        DataContainer<data_t>
            solve(index_t iterations,
                  std::optional<DataContainer<data_t>> x0 = std::nullopt) override;

    protected:
        /// implement the polymorphic clone operation
        auto cloneImpl() const -> ISTA<data_t>* override;

        /// implement the polymorphic comparison operation
        auto isEqual(const Solver<data_t>& other) const -> bool override;

    private:
        /// private constructor called by a public constructor without the step size so that
        /// getLipschitzConstant is called by a LASSOProblem and not by a non-converted Problem
        ISTA(const LASSOProblem<data_t>& lassoProb, data_t epsilon);

        /// The LASSO optimization problem
        LASSOProblem<data_t> _problem;

        /// the default number of iterations
        const index_t _defaultIterations{100};

        /// the step size
        MaybeUninitialized<data_t> _mu;

        /// variable affecting the stopping condition
        data_t _epsilon;
    };
} // namespace elsa
