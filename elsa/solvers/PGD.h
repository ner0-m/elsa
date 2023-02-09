#pragma once

#include <limits>
#include <optional>

#include "DataContainer.h"
#include "LinearOperator.h"
#include "MaybeUninitialized.hpp"
#include "Solver.h"
#include "StrongTypes.h"
#include "LASSOProblem.h"
#include "ProximalOperator.h"

namespace elsa
{
    /**
     * @brief Proximal Gradient Descent (PGD)
     *
     * PGD minimizes function of the form:
     *
     * @f[
     * \min_x g(x) + h(x)
     * @f]
     *
     * where @f$g: \mathbb{R}^n \to \mathbb{R}@f$ is convex and differentiable,
     * and @f$h: \mathbb{R}^n \to \mathbb{R} \cup \{-\infty, \infty\}@f$ is closed
     * convex. Importantly @f$h@f$ needs not to be differentiable, but it needs
     * an proximal operator. Usually, the proximal operator is assumed to be simple,
     * and have an analytical solution.
     *
     * This class currently implements the special case of @f$ g(x) = \frac{1}{2}
     * ||A x - b||_2^2@f$. However, @f$h@f$ can be choosen freely.
     *
     * Given @f$g@f$ defined as above and a convex set @f$\mathcal{C}@f$, one can
     * define an constrained optimization problem:
     * @f[
     * \min_{x \in \mathcal{C}} g(x)
     * @f]
     * Such constraints can take the form of, non-negativity or box constraints.
     * This can be reformulated as an unconstrained problem:
     * @f[
     * \min_{x} g(x) + \mathcal{I}_{\mathcal{C}}(x)
     * @f]
     * where @f$\mathcal{I}_{\mathcal{C}}(x)@f$ is the indicator function of the
     * convex set @f$\mathcal{C}@f$, defined as:
     *
     * @f[
     * \mathcal{I}_{\mathcal{C}}(x) =
     * \begin{cases}
     *     0,    & \text{if } x \in \mathcal{C} \\
     *     \infty, & \text{if } x \notin \mathcal{C}
     * \end{cases}
     * @f]
     *
     * References:
     * -
     * http://www.cs.cmu.edu/afs/cs/Web/People/airg/readings/2012_02_21_a_fast_iterative_shrinkage-thresholding.pdf
     * - https://arxiv.org/pdf/2008.02683.pdf
     *
     * @note PGD has a worst-case complexity result of @f$ O(1/k) @f$.
     *
     * @note A special class of optimization is of the form:
     * @f[
     * \min_{x} \frac{1}{2} || A x - b ||_2^2 + ||x||_1
     * @f]
     * often refered to as @f$\ell_1@f$-Regularization. In this case, the proximal operator
     * for the @f$\ell_1@f$-Regularization is the soft thresolding operator (ProximalL1). This
     * can also be extended with constrains, such as non-negativity constraints.
     *
     * @see An accerlerated version of proximal gradient descent is APGD.
     * See also LASSOProblem
     *
     * @author
     * - Andi Braimllari - initial code
     * - David Frank - generalization
     *
     * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
     */
    template <typename data_t = real_t>
    class PGD : public Solver<data_t>
    {
    public:
        /// Scalar alias
        using Scalar = typename Solver<data_t>::Scalar;

        PGD(const LinearOperator<data_t>& A, const DataContainer<data_t>& b,
            ProximalOperator<data_t> prox, geometry::Threshold<data_t> mu,
            data_t epsilon = std::numeric_limits<data_t>::epsilon());

        PGD(const LinearOperator<data_t>& A, const DataContainer<data_t>& b,
            ProximalOperator<data_t> prox, data_t epsilon = std::numeric_limits<data_t>::epsilon());

        /**
         * @brief Constructor for PGD, accepting a LASSO problem, a fixed step size and optionally,
         * a value for epsilon
         *
         * @param[in] problem the LASSO problem that is supposed to be solved
         * @param[in] mu the fixed step size to be used while solving
         * @param[in] epsilon affects the stopping condition
         */
        PGD(const LASSOProblem<data_t>& problem, geometry::Threshold<data_t> mu,
            data_t epsilon = std::numeric_limits<data_t>::epsilon());

        /**
         * @brief Constructor for PGD, accepting a problem, a fixed step size and optionally, a
         * value for epsilon
         *
         * @param[in] problem the problem that is supposed to be solved
         * @param[in] mu the fixed step size to be used while solving
         * @param[in] epsilon affects the stopping condition
         *
         * Conversion to a LASSOProblem will be attempted. Throws if conversion fails. See
         * LASSOProblem for further details.
         */
        PGD(const Problem<data_t>& problem, geometry::Threshold<data_t> mu,
            data_t epsilon = std::numeric_limits<data_t>::epsilon());

        /**
         * @brief Constructor for PGD, accepting a problem and optionally, a value for
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
        PGD(const Problem<data_t>& problem,
            data_t epsilon = std::numeric_limits<data_t>::epsilon());

        /// make copy constructor deletion explicit
        PGD(const PGD<data_t>&) = delete;

        /// default destructor
        ~PGD() override = default;

        /**
         * @brief Solve the optimization problem, i.e. apply iterations number of iterations of
         * PGD
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
        auto cloneImpl() const -> PGD<data_t>* override;

        /// implement the polymorphic comparison operation
        auto isEqual(const Solver<data_t>& other) const -> bool override;

    private:
        /// private constructor called by a public constructor without the step size so that
        /// getLipschitzConstant is called by a LASSOProblem and not by a non-converted Problem
        PGD(const LASSOProblem<data_t>& lassoProb, data_t epsilon);

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

    template <class data_t = real_t>
    using ISTA = PGD<data_t>;
} // namespace elsa
