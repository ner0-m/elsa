#pragma once

#include "DataContainer.h"
#include "DataDescriptor.h"
#include "Functional.h"

namespace elsa
{
    /// @brief Constant functional. This functinoal maps all input values to a constant scalar
    /// value.
    template <typename data_t = real_t>
    class ConstantFunctional : public Functional<data_t>
    {
    public:
        /// @brief Constructor for the constant functional, mapping domain vector to a scalar
        /// (without a residual)
        ConstantFunctional(const DataDescriptor& descriptor, SelfType_t<data_t> constant);

        /// make copy constructor deletion explicit
        ConstantFunctional(const ConstantFunctional<data_t>&) = delete;

        /// default destructor
        ~ConstantFunctional() override = default;

        bool isDifferentiable() const override;

        /// Return the constant of the functional
        data_t getConstant() const;

    protected:
        /// Return the constant value
        data_t evaluateImpl(const DataContainer<data_t>& Rx) override;

        /// The gradient operator is the ZeroOperator, hence set Rx to 0
        void getGradientImpl(const DataContainer<data_t>& Rx, DataContainer<data_t>& out) override;

        /// There does not exist a hessian, this will throw if called
        LinearOperator<data_t> getHessianImpl(const DataContainer<data_t>& Rx) override;

        /// implement the polymorphic clone operation
        ConstantFunctional<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Functional<data_t>& other) const override;

    private:
        data_t constant_;
    };

    /// @brief Zero functional. This functinoal maps all input values to a zero
    template <typename data_t = real_t>
    class ZeroFunctional : public Functional<data_t>
    {
    public:
        /// @brief Constructor for the zero functional, mapping domain vector to a scalar (without
        /// a residual)
        ZeroFunctional(const DataDescriptor& descriptor);

        /// make copy constructor deletion explicit
        ZeroFunctional(const ConstantFunctional<data_t>&) = delete;

        bool isDifferentiable() const override;

        /// default destructor
        ~ZeroFunctional() override = default;

    protected:
        /// Return the constant value
        data_t evaluateImpl(const DataContainer<data_t>& Rx) override;

        /// The gradient operator is the ZeroOperator, hence set Rx to 0
        void getGradientImpl(const DataContainer<data_t>& Rx, DataContainer<data_t>& out) override;

        /// There does not exist a hessian, this will throw if called
        LinearOperator<data_t> getHessianImpl(const DataContainer<data_t>& Rx) override;

        /// implement the polymorphic clone operation
        ZeroFunctional<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Functional<data_t>& other) const override;
    };

} // namespace elsa
