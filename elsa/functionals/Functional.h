#pragma once

#include "Cloneable.h"
#include "DataDescriptor.h"
#include "Residual.h"
#include "LinearOperator.h"

namespace elsa
{
    /**
     * @brief Abstract base class representing a functional, i.e. a mapping from vectors to scalars.
     *
     * @author Matthias Wieczorek - initial code
     * @author Maximilian Hornung - modularization
     * @author Tobias Lasser - rewrite
     *
     * @tparam data_t data type for the domain of the residual of the functional, defaulting to
     * real_t
     *
     * A functional is a mapping a vector to a scalar value (e.g. mapping the output of a Residual
     * to a scalar). Typical examples of functionals are norms or semi-norms, such as the L2 or L1
     * norms.
     *
     * Using LinearOperators, Residuals (e.g. LinearResidual) and a Functional (e.g. L2NormPow2)
     * enables the formulation of typical terms in an OptimizationProblem.
     */
    template <typename data_t = real_t>
    class Functional : public Cloneable<Functional<data_t>>
    {
    public:
        /**
         * @brief Constructor for the functional, mapping a domain vector to a scalar (without a
         * residual)
         *
         * @param[in] domainDescriptor describing the domain of the functional
         */
        explicit Functional(const DataDescriptor& domainDescriptor);

        /**
         * @brief Constructor for the functional, using a Residual as input to map to a scalar
         *
         * @param[in] residual to be used when evaluating the functional (or its derivatives)
         */
        explicit Functional(const Residual<data_t>& residual);

        /// default destructor
        ~Functional() override = default;

        /// return the domain descriptor
        const DataDescriptor& getDomainDescriptor() const;

        /// return the residual (will be trivial if Functional was constructed without one)
        const Residual<data_t>& getResidual() const;

        /**
         * @brief evaluate the functional at x and return the result
         *
         * @param[in] x input DataContainer (in the domain of the functional)
         *
         * @returns result the scalar of the functional evaluated at x
         *
         * Please note: after evaluating the residual at x, this method calls the method
         * evaluateImpl that has to be overridden in derived classes to compute the functional's
         * value.
         */
        data_t evaluate(const DataContainer<data_t>& x);

        /**
         * @brief compute the gradient of the functional at x and return the result
         *
         * @param[in] x input DataContainer (in the domain of the functional)
         *
         * @returns result DataContainer (in the domain of the functional) containing the result of
         * the gradient at x.
         *
         * Please note: this method uses getGradient(x, result) to perform the actual operation.
         */
        DataContainer<data_t> getGradient(const DataContainer<data_t>& x);

        /**
         * @brief compute the gradient of the functional at x and store in result
         *
         * @param[in] x input DataContainer (in the domain of the functional)
         * @param[out] result output DataContainer (in the domain of the functional)
         *
         * Please note: after evaluating the residual at x, this methods calls the method
         * getGradientInPlaceImpl that has to be overridden in derived classes to compute the
         * functional's gradient, and after that the chain rule for the residual is applied (if
         * necessary).
         */
        void getGradient(const DataContainer<data_t>& x, DataContainer<data_t>& result);

        /**
         * @brief return the Hessian of the functional at x
         *
         * @param[in] x input DataContainer (in the domain of the functional)
         *
         * @returns a LinearOperator (the Hessian)
         *
         * Note: some derived classes might decide to use only the diagonal of the Hessian as a fast
         * approximation!
         *
         * Please note: after evaluating the residual at x, this method calls the method
         * getHessianImpl that has to be overridden in derived classes to compute the functional's
         * Hessian, and after that the chain rule for the residual is applied (if necessary).
         */
        std::unique_ptr<LinearOperator<data_t>> getHessian(const DataContainer<data_t>& x);

    protected:
        /// the data descriptor of the domain of the functional
        std::unique_ptr<DataDescriptor> _domainDescriptor;

        /// the residual
        std::unique_ptr<Residual<data_t>> _residual;

        /// implement the polymorphic comparison operation
        bool isEqual(const Functional<data_t>& other) const override;

        /**
         * @brief the evaluateImpl method that has to be overridden in derived classes
         *
         * @param[in] Rx the residual evaluated at x
         *
         * @returns the evaluated functional
         *
         * Please note: the evaluation of the residual is already performed in evaluate, so this
         * method only has to compute the functional's value itself.
         */
        virtual data_t evaluateImpl(const DataContainer<data_t>& Rx) = 0;

        /**
         * @brief the getGradientInPlaceImpl method that has to be overridden in derived classes
         *
         * @param[in,out] Rx the residual evaluated at x (in), and the gradient of the functional
         * (out)
         *
         * Please note: the evaluation of the residual is already performed in getGradient, as well
         * as the application of the chain rule. This method here only has to compute the gradient
         * of the functional itself, in an in-place manner (to avoid unnecessary DataContainers).
         */
        virtual void getGradientInPlaceImpl(DataContainer<data_t>& Rx) = 0;

        /**
         * @brief the getHessianImpl method that has to be overridden in derived classes
         *
         * @param[in] Rx the residual evaluated at x
         *
         * @returns the LinearOperator representing the Hessian of the functional
         *
         * Please note: the evaluation of the residual is already performed in getHessian, as well
         * as the application of the chain rule. This method here only has to compute the Hessian of
         * the functional itself.
         */
        virtual std::unique_ptr<LinearOperator<data_t>>
            getHessianImpl(const DataContainer<data_t>& Rx) = 0;
    };
} // namespace elsa
