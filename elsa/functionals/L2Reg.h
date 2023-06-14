#pragma once

#include "DataDescriptor.h"
#include "Functional.h"
#include "DataContainer.h"
#include "LinearOperator.h"

namespace elsa
{
    /**
     * @brief Class representing a L2 regularization term with an optional linear operator
     *
     * This functional evaluates to \f$ 0.5 * || A(x) ||_2^2 \f$. The L2Reg should be used
     * if an L2Squared norm is not sufficient as an linear operator is necessary.
     *
     * Note, that the proximal operator is not analytically computable in this case.
     *
     * @tparam data_t data type for the domain of the residual of the functional, defaulting to
     * real_t
     *
     * @see L2Squared
     */
    template <typename data_t = real_t>
    class L2Reg : public Functional<data_t>
    {
    public:
        /**
         * @brief Constructor the l2 regularization functional without data and linear operator,
         * i.e. \f$A = I\f$ and \f$b = 0\f$.
         *
         * @param[in] domainDescriptor describing the domain of the functional
         */
        explicit L2Reg(const DataDescriptor& domain);

        /**
         * @brief Constructor the l2 regularization functional with an linear operator
         *
         * @param[in] A linear operator to be used
         */
        explicit L2Reg(const LinearOperator<data_t>& A);

        /// make copy constructor deletion explicit
        L2Reg(const L2Reg<data_t>&) = delete;

        /// default destructor
        ~L2Reg() override = default;

        bool hasOperator() const;

        const LinearOperator<data_t>& getOperator() const;

        bool isDifferentiable() const override;

    protected:
        /// the evaluation of the l2 norm (squared)
        data_t evaluateImpl(const DataContainer<data_t>& Rx) override;

        /// the computation of the gradient (in place)
        void getGradientImpl(const DataContainer<data_t>& Rx, DataContainer<data_t>& out) override;

        /// the computation of the Hessian
        LinearOperator<data_t> getHessianImpl(const DataContainer<data_t>& Rx) override;

        /// implement the polymorphic clone operation
        L2Reg<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Functional<data_t>& other) const override;

    private:
        std::unique_ptr<LinearOperator<data_t>> A_{};
    };

} // namespace elsa
