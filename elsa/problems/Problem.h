#pragma once

#include "Functional.h"
#include "RegularizationTerm.h"

#include <vector>

namespace elsa
{
    template <typename data_t>
    class LASSOProblem;

    /**
     * @brief Class representing a generic optimization problem consisting of data term and
     * regularization term(s).
     *
     * @author Matthias Wieczorek - initial code
     * @author Maximilian Hornung - modularization
     * @author Tobias Lasser - rewrite, modernization
     *
     * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
     *
     * This class represents a generic optimization problem, which consists of a data term and
     * (optionally) of one (or many) regularization terms, \f$ \argmin_x D(x) + \sum_{i=1}^n
     * \lambda_i R(x) \f$. Here, the data term \f$ D(x) \f$ is represented through a Functional (or
     * it derivatives), the regularization terms are represented by RegularizationTerms, which
     * encapsulate regularization parameters \f$ \lambda_i \f$ (scalar values) and the actual
     * regularization terms \f$ R(x) \f$ (Functionals or its derivatives).
     */
    template <typename data_t = real_t>
    class Problem : public Cloneable<Problem<data_t>>
    {
    public:
        /**
         * @brief Constructor for optimization problem, accepting a data and multiple regularization
         * terms, and an initial guess x0.
         *
         * @param[in] dataTerm functional expressing the data term
         * @param[in] regTerms vector of RegularizationTerms (weight and functional)
         * @param[in] x0 initial value for the current estimated solution
         * @param[in] lipschitzConstant if non-null the known lipschitz constant of the
         * problem. If null the lipschitz constant will be computed using power-iteration. Useful in
         * cases where the numerical approximation is not accurate and the constant is known.
         */
        Problem(const Functional<data_t>& dataTerm,
                const std::vector<RegularizationTerm<data_t>>& regTerms,
                const DataContainer<data_t>& x0, std::optional<data_t> lipschitzConstant = {});

        /**
         * @brief Constructor for optimization problem, accepting a data and multiple regularization
         * terms.
         *
         * @param[in] dataTerm functional expressing the data term
         * @param[in] regTerms vector of RegularizationTerms (weight and functional)
         * @param[in] lipschitzConstant if non-null the known lipschitz constant of the
         * problem. If null the lipschitz constant will be computed using power-iteration. Useful in
         * cases where the numerical approximation is not accurate and the constant is known.
         */
        Problem(const Functional<data_t>& dataTerm,
                const std::vector<RegularizationTerm<data_t>>& regTerms,
                std::optional<data_t> lipschitzConstant = {});

        /**
         * @brief Constructor for optimization problem, accepting a data and one regularization
         * term, and an initial guess x0.
         *
         * @param[in] dataTerm functional expressing the data term
         * @param[in] regTerm RegularizationTerm (weight and functional)
         * @param[in] x0 initial value for the current estimated solution
         * @param[in] lipschitzConstant if non-null the known lipschitz constant of the
         * problem. If null the lipschitz constant will be computed using power-iteration. Useful in
         * cases where the numerical approximation is not accurate and the constant is known.
         */
        Problem(const Functional<data_t>& dataTerm, const RegularizationTerm<data_t>& regTerm,
                const DataContainer<data_t>& x0, std::optional<data_t> lipschitzConstant = {});

        /**
         * @brief Constructor for optimization problem, accepting a data and one regularization
         * term.
         *
         * @param[in] dataTerm functional expressing the data term
         * @param[in] regTerm RegularizationTerm (weight and functional)
         * @param[in] lipschitzConstant if non-null the known lipschitz constant of the
         * problem. If null the lipschitz constant will be computed using power-iteration. Useful in
         * cases where the numerical approximation is not accurate and the constant is known.
         */
        Problem(const Functional<data_t>& dataTerm, const RegularizationTerm<data_t>& regTerm,
                std::optional<data_t> lipschitzConstant = {});

        /**
         * @brief Constructor for optimization problem, accepting a data term and an initial guess
         * x0.
         *
         * @param[in] dataTerm functional expressing the data term
         * @param[in] x0 initial value for the current estimated solution
         * @param[in] lipschitzConstant if non-null the known lipschitz constant of the
         * problem. If null the lipschitz constant will be computed using power-iteration. Useful in
         * cases where the numerical approximation is not accurate and the constant is known.
         */
        Problem(const Functional<data_t>& dataTerm, const DataContainer<data_t>& x0,
                std::optional<data_t> lipschitzConstant = {});

        /**
         * @brief Constructor for optimization problem, accepting a data term.
         *
         * @param[in] dataTerm functional expressing the data term
         * @param[in] lipschitzConstant if non-null the known lipschitz constant of the
         * problem. If null the lipschitz constant will be computed using power-iteration. Useful in
         * cases where the numerical approximation is not accurate and the constant is known.
         */
        explicit Problem(const Functional<data_t>& dataTerm,
                         std::optional<data_t> lipschitzConstant = {});

        /// default destructor
        ~Problem() override = default;

        /// return the data term
        const Functional<data_t>& getDataTerm() const;

        /// return the vector of regularization terms
        const std::vector<RegularizationTerm<data_t>>& getRegularizationTerms() const;

        /// return the current estimated solution (const version)
        const DataContainer<data_t>& getCurrentSolution() const;

        /// return the current estimated solution
        DataContainer<data_t>& getCurrentSolution();

        /**
         * @brief evaluate the problem at the current estimated solution
         *
         * @returns the value of the problem evaluated at the current estimated solution
         *
         * Please note: this method calls the method evaluateImpl that has to be overridden in
         * derived classes.
         */
        data_t evaluate();

        /**
         * @brief return the gradient of the problem at the current estimated solution
         *
         * @returns DataContainer (in the domain of the problem) containing the result of the
         * gradient at the current solution
         *
         * Please note: this method used getGradient(result) to perform the actual operation.
         */
        DataContainer<data_t> getGradient();

        /**
         * @brief compute the gradient of the problem at the current estimated solution
         *
         * @param[out] result output DataContainer containing the gradient (in the domain of the
         * problem)
         *
         * Please note: this method calls the method getGradientImpl that has to be overridden in
         * derived classes.
         */
        void getGradient(DataContainer<data_t>& result);

        /**
         * @brief return the Hessian of the problem at the current estimated solution
         *
         * @returns a LinearOperator (the Hessian)
         *
         * Please note: this method calls the method getHessianImpl that has to be overridden in
         * derived classes.
         */
        LinearOperator<data_t> getHessian() const;

        /**
         * @brief return the Lipschitz Constant of the problem at the current estimated solution. If
         * an explicit lipschitz constant has been passed to the problem it will be returned here.
         *
         * @param[in] nIterations number of iterations to compute the lipschitz constant using
         * power iteration.
         *
         * @returns data_t (the Lipschitz Constant)
         *
         * Please note: this method calls the method getLipschitzConstantImpl that has to be
         * overridden in derived classes which want to provide a more specific way of computing
         * the Lipschitz constant, e.g. by not using power iteration or where the hessian is already
         * approximated as a diagonal matrix.
         */
        data_t getLipschitzConstant(index_t nIterations = 5) const;

        operator LASSOProblem<data_t>() const;

    protected:
        /// the data term
        std::unique_ptr<Functional<data_t>> _dataTerm{};

        /// the regularization terms
        std::vector<RegularizationTerm<data_t>> _regTerms{};

        /// the current estimated solution
        DataContainer<data_t> _currentSolution;

        /// the known lipschitz constant for a problem, if not given will be computed on demand
        std::optional<data_t> _lipschitzConstant = {};

        /// protected copy constructor, simplifies cloning (of the subclasses primarily)
        Problem(const Problem<data_t>& problem);

        /// the evaluation of the optimization problem
        virtual data_t evaluateImpl();

        /// the getGradient method for the optimization problem
        virtual void getGradientImpl(DataContainer<data_t>& result);

        /// the getHessian method for the optimization problem
        virtual LinearOperator<data_t> getHessianImpl() const;

        /// the getLipschitzConstant method for the optimization problem
        virtual data_t getLipschitzConstantImpl(index_t nIterations) const;

        /// implement the polymorphic clone operation
        Problem<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Problem<data_t>& other) const override;
    };
} // namespace elsa
