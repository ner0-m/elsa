#pragma once

#include "elsa.h"
#include "Cloneable.h"
#include "DataContainer.h"
#include "DataDescriptor.h"
#include "LinearOperator.h"

namespace elsa
{
    /**
     * \brief Base class representing an optimization problem.
     *
     * \author Mathias Wieczorek - initial code
     * \author Maximilian Hornung - modularization
     * \author Tobias Lasser - rewrite, modernization
     *
     * \tparam data_t data type for the domain and range of the problem, defaulting to real_t
     *
     * This class represents an abstract optimization problem, for use with iterative solvers.
     * It stores a current estimated solution.
     */
    template <typename data_t = real_t>
    class Problem : public Cloneable<Problem<data_t>> {
    public:
        /**
         * \brief Constructor for the problem, accepting an initial guess x0
         *
         * \param[in] x0 initial value for the current estimated solution variable
         */
        explicit Problem(const DataContainer<data_t>& x0);

        /**
         * \brief Constructor for the problem (initial guess implicitly set to 0)
         *
         * \param[in] domainDescriptor describing the domain of the problem
         */
        explicit Problem(const DataDescriptor& domainDescriptor);

        /// default destructor
        ~Problem() override = default;

        /// return the current estimated solution (const version)
        const DataContainer<data_t>& getCurrentSolution() const;

        /// return the current estimated solution
        DataContainer<data_t>& getCurrentSolution();


        /**
         * \brief evaluate the problem at the current estimated solution
         *
         * \returns the value of the problem evaluated at the current estimated solution
         *
         * Please note: this method calls the method _evaluate that has to be overridden in derived classes.
         */
        data_t evaluate();

        /**
         * \brief return the gradient of the problem at the current estimated solution
         *
         * \returns DataContainer (in the domain of the problem) containing the result of the
         * gradient at the current solution
         *
         * Please note: this method used getGradient(result) to perform the actual operation.
         */
        DataContainer<data_t> getGradient();

        /**
         * \brief compute the gradient of the problem at the current estimated solution
         *
         * \param[out] result output DataContainer containing the gradient (in the domain of the problem)
         *
         * Please note: this method calls the method _getGradient that has to be overridden in derived classes.
         */
        void getGradient(DataContainer<data_t>& result);

        /**
         * \brief return the Hessian of the problem at the current estimated solution
         *
         * \returns a LinearOperator (the Hessian)
         *
         * Please note: this method calls the method _getHessian that has to be overridden in derived classes.
         */
        LinearOperator<data_t> getHessian();



    protected:
        /// the current estimated solution
        DataContainer<data_t> _currentSolution;

        /// the evaluate method that has to be overridden in derived classes
        virtual data_t _evaluate() = 0;

        /// the getGradient method that has to be overridden in derived classes
        virtual void _getGradient(DataContainer<data_t>& result) = 0;

        /// the getHessian method that has to be overridden in derived classes
        virtual LinearOperator<data_t> _getHessian() = 0;

        /// implement the polymorphic comparison operation
        bool isEqual(const Problem<data_t>& other) const override;
    };
} // namespace elsa
