#pragma once

#include "Constraint.h"
#include "L0PseudoNorm.h"
#include "L1Norm.h"
#include "WLSProblem.h"
#include "LASSOProblem.h"

namespace elsa
{
    /**
     * @brief Class representing a splitting problem
     *
     * @author Andi Braimllari - initial code
     *
     * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
     *
     * This class represents a splitting problem i.e.
     *
     *  - minimize @f$ f(x) + g(z) @f$ subject to @f$ Ax + Bz = c @f$.
     *
     * in which @f$ A @f$ and @f$ B @f$ are linear operators, @f$ c @f$ is a data vector.
     *
     * References:
     * https://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
     */
    template <typename data_t = real_t>
    class SplittingProblem : public Problem<data_t>
    {
    public:
        /**
         * @brief Constructor for SplittingProblem, accepting a functional, regularization terms and
         * a constraint
         *
         * @param[in] f the functional from the problem statement
         * @param[in] g the regularization terms from the problem statement
         * @param[in] constraint the constraint from the problem statement
         */
        SplittingProblem(const Functional<data_t>& f,
                         const std::vector<RegularizationTerm<data_t>>& g,
                         const Constraint<data_t>& constraint);

        /**
         * @brief Constructor for SplittingProblem, accepting a functional, a regularization term
         * and a constraint
         *
         * @param[in] f the functional from the problem statement
         * @param[in] g the regularization term from the problem statement
         * @param[in] constraint the constraint from the problem statement
         */
        SplittingProblem(const Functional<data_t>& f, const RegularizationTerm<data_t>& g,
                         const Constraint<data_t>& constraint);

        /// default destructor
        ~SplittingProblem() override = default;

        /// return the constraint
        auto getConstraint() const -> const Constraint<data_t>&;

        /// return the f problem
        auto getF() const -> const Functional<data_t>&;

        /// return the g problem
        auto getG() const -> const std::vector<RegularizationTerm<data_t>>&;

    protected:
        /// implement the polymorphic clone operation
        auto cloneImpl() const -> SplittingProblem<data_t>* override;

        /// the evaluation method of the splitting problem
        auto evaluateImpl() -> data_t override;

        /// the getGradient method for the splitting problem
        void getGradientImpl(DataContainer<data_t>& result) override;

        /// the getHessian method for the splitting problem
        auto getHessianImpl() const -> LinearOperator<data_t> override;

        /// the getLipschitzConstant method for the splitting problem
        auto getLipschitzConstantImpl(index_t nIterations) const -> data_t override;

    private:
        std::unique_ptr<Functional<data_t>> _f;

        std::vector<RegularizationTerm<data_t>> _g;

        std::unique_ptr<Constraint<data_t>> _constraint;
    };
} // namespace elsa
