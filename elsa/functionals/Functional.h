#pragma once

#include "Cloneable.h"
#include "DataDescriptor.h"
#include "Error.h"
#include "Residual.h"
#include "LinearOperator.h"
#include "TypeCasts.hpp"
#include "elsaDefines.h"

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
        LinearOperator<data_t> getHessian(const DataContainer<data_t>& x);

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
        virtual LinearOperator<data_t> getHessianImpl(const DataContainer<data_t>& Rx) = 0;
    };

    /**
     * @brief Class representing a sum of two functionals
     * \f[
     * f(x) = h(x) + g(x)
     * \f]
     *
     * The gradient at \f$x\f$ is given as:
     * \f[
     * \nabla f(x) = \nabla h(x) + \nabla g(x)
     * \f]
     *
     * and finally the hessian is given by:
     * \f[
     * \nabla^2 f(x) = \nabla^2 h(x) \nabla^2 g(x)
     * \f]
     *
     * The gradient and hessian is only valid if the functional is (twice)
     * differentiable. The `operator+` is overloaded for, to conviniently create
     * this class. It should not be necessary to create it explicitly.
     */
    template <class data_t>
    class FunctionalSum final : public Functional<data_t>
    {
    public:
        /// Construct from two functionals
        FunctionalSum(const Functional<data_t>& lhs, const Functional<data_t>& rhs);

        /// Make deletion of copy constructor explicit
        FunctionalSum(const FunctionalSum<data_t>&) = delete;

        /// Make deleteion of copy assignment explicit
        FunctionalSum& operator=(const FunctionalSum<data_t>&) = delete;

    private:
        /// evaluate the functional as \f$g(x) + h(x)\f$
        data_t evaluateImpl(const DataContainer<data_t>& Rx) override;

        /// evaluate the gradient as: \f$\nabla g(x) + \nabla h(x)\f$
        void getGradientInPlaceImpl(DataContainer<data_t>& Rx) override;

        /// construct the hessian as: \f$\nabla^2 g(x) + \nabla^2 h(x)\f$
        LinearOperator<data_t> getHessianImpl(const DataContainer<data_t>& Rx) override;

        /// Implement polymorphic clone
        FunctionalSum<data_t>* cloneImpl() const override;

        /// Implement polymorphic equality
        bool isEqual(const Functional<data_t>& other) const override;

        /// Store the left hand side functionl
        std::unique_ptr<Functional<data_t>> lhs_{};

        /// Store the right hand side functional
        std::unique_ptr<Functional<data_t>> rhs_{};
    };

    /**
     * @brief Class representing a functional with a scalar multiplication:
     * \f[
     * f(x) = \lambda * g(x)
     * \f]
     *
     * The gradient at \f$x\f$ is given as:
     * \f[
     * \nabla f(x) = \lambda \nabla g(x)
     * \f]
     *
     * and finally the hessian is given by:
     * \f[
     * \nabla^2 f(x) = \lambda \nabla^2 g(x)
     * \f]
     *
     * The gradient and hessian is only valid if the functional is differentiable.
     * The `operator*` is overloaded for scalar values with functionals, to
     * conviniently create this class. It should not be necessary to create it
     * explicitly.
     */
    template <class data_t>
    class FunctionalScalarMul final : public Functional<data_t>
    {
    public:
        /// Construct functional from other functional and scalar
        FunctionalScalarMul(const Functional<data_t>& fn, SelfType_t<data_t> scalar);

        /// Make deletion of copy constructor explicit
        FunctionalScalarMul(const FunctionalScalarMul<data_t>&) = delete;

        /// Make deletion of copy assignment explicit
        FunctionalScalarMul& operator=(const FunctionalScalarMul<data_t>&) = delete;

        ~FunctionalScalarMul() override = default;

    private:
        /// Evaluate as \f$\lambda * \nabla g(x)\f$
        data_t evaluateImpl(const DataContainer<data_t>& Rx) override;

        /// Evaluate gradient as: \f$\lambda * \nabla g(x)\f$
        void getGradientInPlaceImpl(DataContainer<data_t>& Rx) override;

        /// Construct hessian as: \f$\lambda * \nabla^2 g(x)\f$
        LinearOperator<data_t> getHessianImpl(const DataContainer<data_t>& Rx) override;

        /// Implementation of polymorphic clone
        FunctionalScalarMul<data_t>* cloneImpl() const override;

        /// Implementation of polymorphic equality
        bool isEqual(const Functional<data_t>& other) const override;

        /// Store other functional \f$g\f$
        std::unique_ptr<Functional<data_t>> fn_{};

        /// The scalar
        data_t scalar_;
    };

    template <class data_t>
    FunctionalScalarMul<data_t> operator*(SelfType_t<data_t> s, const Functional<data_t>& f)
    {
        // TODO: consider returning the ZeroFunctional, if s == 0, but then
        // it's necessary to return unique_ptr and I hate that
        return FunctionalScalarMul<data_t>(f, s);
    }

    template <class data_t>
    FunctionalScalarMul<data_t> operator*(const Functional<data_t>& f, SelfType_t<data_t> s)
    {
        // TODO: consider returning the ZeroFunctional, if s == 0, but then
        // it's necessary to return unique_ptr and I hate that
        return FunctionalScalarMul<data_t>(f, s);
    }

    template <class data_t>
    FunctionalSum<data_t> operator+(const Functional<data_t>& lhs, const Functional<data_t>& rhs)
    {
        return FunctionalSum<data_t>(lhs, rhs);
    }

} // namespace elsa
