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
     * The indicator function evaluates to @f$ 0 @f$ if the input is @f$ \geq 0 @f$, @f$ +\infty @f$
     * otherwise.
     *
     * TODO ideally we should be able to define the set here instead of simply >=0, do we have this
     *  in elsa?
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
         */
        explicit Indicator(const DataDescriptor& domainDescriptor);

        // TODO add Residual constructor?

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
    };
} // namespace elsa
