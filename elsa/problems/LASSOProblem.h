#pragma once

#include "L1Norm.h"
#include "WLSProblem.h"
#include "LinearOperator.h"
#include "DataContainer.h"

namespace elsa
{
    /**
     * @brief Class representing a Least Absolute Shrinkage and Selection Operator problem
     *
     * This class represents a LASSO problem i.e.
     *
     *  - @f$ \argmin_x \frac{1}{2} \| Ax - b \|_{W,2}^2 + \lambda \| x \|_1 @f$
     *
     * in which @f$ W @f$ is a weighting (scaling) operator, @f$ A @f$ is a linear operator, @f$
     * b @f$ is a data vector and @f$ \lambda @f$ is the regularization parameter.
     *
     * References:
     * - Ryan J. Tibshirani _The Lasso Problem and Uniqueness_ (2013)
     *   https://www.stat.cmu.edu/~ryantibs/papers/lassounique.pdf
     * - Tao, S., Boley, D., Zhang, S. _Local Linear Convergence of ISTA and FISTA on the LASSO
     *   Problem_ (2015) https://arxiv.org/pdf/1501.02888.pdf
     *
     * @author Andi Braimllari - initial code
     *
     * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
     */
    template <typename data_t = real_t>
    class LASSOProblem : public Problem<data_t>
    {
    public:
        /**
         * @brief Constructor for the lasso problem, construction a WLSProblem
         *
         * @param[in] A a linear operator
         * @param[in] b a data vector
         * @param[in] regTerm RegularizationTerm
         */
        LASSOProblem(const LinearOperator<data_t>& A, const DataContainer<data_t>& b,
                     real_t lambda = 0.5f);

        /**
         * @brief Constructor for the lasso problem, accepting wlsProblem and regTerm
         *
         * @param[in] wlsProblem WLSProblem
         * @param[in] regTerm RegularizationTerm
         */
        LASSOProblem(WLSProblem<data_t> wlsProblem, const RegularizationTerm<data_t>& regTerm);

        /**
         * @brief Constructor for converting a general optimization problem to a LASSO one
         *
         * @param[in] problem the problem to be converted
         *
         * Only problems that consist exclusively of a WLSProblem and a L1Norm regularization term
         * can be converted.
         *
         * Acts as a copy constructor if the supplied optimization problem is a LASSO problem.
         */
        explicit LASSOProblem(const Problem<data_t>& problem);

        /// default destructor
        ~LASSOProblem() override = default;

    protected:
        /// implement the polymorphic clone operation
        auto cloneImpl() const -> LASSOProblem<data_t>* override;

        /// the getLipschitzConstant method for the optimization problem
        auto getLipschitzConstantImpl(index_t nIterations) const -> data_t override;

    private:
        WLSProblem<data_t> _wlsProblem;

        /// converts an optimization problem to a WLSProblem
        static auto wlsFromProblem(const Problem<data_t>& problem) -> WLSProblem<data_t>;

        /// converts an optimization problem to a RegularizationTerm
        static auto regTermFromProblem(const Problem<data_t>& problem)
            -> RegularizationTerm<data_t>;
    };
} // namespace elsa
