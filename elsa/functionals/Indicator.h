#pragma once

#include "Functional.h"

namespace elsa
{
    /**
     * @brief Class representing the indicator functional.
     *
     * @author Andi Braimllari - initial code
     *
     * @tparam data_t data type for the domain of the residual of the functional, defaulting to
     * real_t
     *
     * The indicator function evaluates to @f$ 0 @f$ if the input satisfies the constraint, @f$
     * +\infty @f$ otherwise. The constraint is built by the constraint operator (comparison
     * operators) and the constraint value.
     *
     * References:
     * https://arxiv.org/pdf/0912.3522.pdf
     */
    template <typename data_t = real_t>
    class Indicator : public Functional<data_t>
    {
    public:
        /**
         * @brief Constructor for the indicator functional, mapping domain vector to a scalar
         * (without a residual)
         *
         * @param[in] domainDescriptor describing the domain of the functional
         * @param[in] constraintOperation describing the constraint (comparison) operation
         * @param[in] constraintValue describing the value to be used in the constraint
         */
        template <typename Comparator = std::greater_equal<data_t>>
        Indicator(const DataDescriptor& domainDescriptor, Comparator comparator = Comparator{},
                  data_t constraintValue = 0)
            : Functional<data_t>(domainDescriptor),
              _comparator{comparator},
              _constraintValue{constraintValue}
        {
        }

        /// make copy constructor deletion explicit
        Indicator(const Indicator<data_t>&) = delete;

        /// default destructor
        ~Indicator() override = default;

    protected:
        /// the evaluation of the indicator function
        data_t evaluateImpl(const DataContainer<data_t>& Rx) override;

        /// the computation of the gradient (in place)
        void getGradientInPlaceImpl(DataContainer<data_t>& Rx) override;

        /// the computation of the Hessian
        LinearOperator<data_t> getHessianImpl(const DataContainer<data_t>& Rx) override;

        /// implement the polymorphic clone operation
        Indicator<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Functional<data_t>& other) const override;

    private:
        bool constraintIsSatisfied(data_t data);

        std::function<bool(data_t&, data_t&)> _comparator;
        data_t _constraintValue;
    };
} // namespace elsa
