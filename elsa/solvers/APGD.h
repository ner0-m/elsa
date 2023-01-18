#pragma once

#include <optional>

#include "Solver.h"
#include "StrongTypes.h"
#include "LASSOProblem.h"
#include "MaybeUninitialized.hpp"
#include "ProximalOperator.h"

namespace elsa
{
    /**
     * @brief Class representing a Fast Iterative Shrinkage-Thresholding Algorithm solver
     *
     * This class represents a APGD solver i.e.
     *
     *  - @f$ x_{k} = shrinkageOperator(y_k - \mu * A^T (Ay_k - b)) @f$
     *  - @f$ t_{k+1} = \frac{1 + \sqrt{1 + 4 * t_{k}^2}}{2} @f$
     *  - @f$ y_{k+1} = x_{k} + (\frac{t_{k} - 1}{t_{k+1}}) * (x_{k} - x_{k - 1}) @f$
     *
     * in which shrinkageOperator is the proximal operator of the L1-norm, which is
     * often referred to as soft thresholding, defined as @f$
     * shrinkageOperator(z_k) = sign(z_k)·(|z_k| - \mu*\lambda)_+ @f$.
     *
     * APGD has a worst-case complexity result of @f$ O(1/k^2) @f$.
     *
     * References:
     * http://www.cs.cmu.edu/afs/cs/Web/People/airg/readings/2012_02_21_a_fast_iterative_shrinkage-thresholding.pdf
     * https://arxiv.org/pdf/2008.02683.pdf
     *
     * @author
     * Andi Braimllari - initial code
     * David Frank - generalization to APGD
     *
     * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
     */
    template <typename data_t = real_t>
    class APGD : public Solver<data_t>
    {
    public:
        /// Scalar alias
        using Scalar = typename Solver<data_t>::Scalar;

        APGD(const LinearOperator<data_t>& A, const DataContainer<data_t>& b,
             ProximalOperator<data_t> prox, geometry::Threshold<data_t> mu,
             data_t epsilon = std::numeric_limits<data_t>::epsilon());

        APGD(const LinearOperator<data_t>& A, const DataContainer<data_t>& b,
             ProximalOperator<data_t> prox,
             data_t epsilon = std::numeric_limits<data_t>::epsilon());

        /**
         * @brief Constructor for APGD, accepting a LASSO problem, a fixed step size and
         * optionally, a value for epsilon
         *
         * @param[in] problem the LASSO problem that is supposed to be solved
         * @param[in] mu the fixed step size to be used while solving
         * @param[in] epsilon affects the stopping condition
         */
        APGD(const LASSOProblem<data_t>& problem, geometry::Threshold<data_t> mu,
             data_t epsilon = std::numeric_limits<data_t>::epsilon());

        /**
         * @brief Constructor for APGD, accepting a problem, a fixed step size and optionally, a
         * value for epsilon
         *
         * @param[in] problem the problem that is supposed to be solved
         * @param[in] mu the fixed step size to be used while solving
         * @param[in] epsilon affects the stopping condition
         *
         * Conversion to a LASSOProblem will be attempted. Throws if conversion fails. See
         * LASSOProblem for further details.
         */
        APGD(const Problem<data_t>& problem, geometry::Threshold<data_t> mu,
             data_t epsilon = std::numeric_limits<data_t>::epsilon());

        /**
         * @brief Constructor for APGD, accepting a problem and optionally, a value for
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
        APGD(const Problem<data_t>& problem,
             data_t epsilon = std::numeric_limits<data_t>::epsilon());

        /// make copy constructor deletion explicit
        APGD(const APGD<data_t>&) = delete;

        /// default destructor
        ~APGD() override = default;

        /**
         * @brief Solve the optimization problem, i.e. apply iterations number of iterations of
         * APGD
         *
         * @param[in] iterations number of iterations to execute
         *
         * @returns the approximated solution
         */
        DataContainer<data_t>
            solve(index_t iterations,
                  std::optional<DataContainer<data_t>> x0 = std::nullopt) override;

    protected:
        /// implement the polymorphic clone operation
        auto cloneImpl() const -> APGD<data_t>* override;

        /// implement the polymorphic comparison operation
        auto isEqual(const Solver<data_t>& other) const -> bool override;

    private:
        /// private constructor called by a public constructor without the step size so that
        /// getLipschitzConstant is called by a LASSOProblem and not by a non-converted Problem
        APGD(const LASSOProblem<data_t>& lassoProb, data_t epsilon);

        /// The LASSO optimization problem
        std::unique_ptr<LinearOperator<data_t>> A_;

        DataContainer<data_t> b_;

        ProximalOperator<data_t> prox_;

        /// variable affecting the stopping condition
        data_t lambda_;

        /// the step size
        MaybeUninitialized<data_t> mu_;

        /// variable affecting the stopping condition
        data_t epsilon_;
    };
} // namespace elsa
