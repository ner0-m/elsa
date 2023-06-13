#pragma once

#include "DataContainer.h"
#include "Functional.h"

namespace elsa
{
    /**
     * @brief Class representing the l1 norm functional.
     *
     * The l1 norm functional evaluates to \f$ \sum_{i=1}^n |x_i| \f$ for \f$ x=(x_i)_{i=1}^n \f$.
     * Please note that it is not differentiable, hence getGradient and getHessian will throw
     * exceptions.
     *
     * @tparam data_t data type for the domain of the residual of the functional, defaulting to
     * real_t
     *
     * @author
     * * Matthias Wieczorek - initial code
     * * Maximilian Hornung - modularization
     * * Tobias Lasser - modernization
     *
     */
    template <typename data_t = real_t>
    class L1Norm : public Functional<data_t>
    {
    public:
        /**
         * @brief Constructor for the l1 norm functional, mapping domain vector to a scalar (without
         * a residual)
         *
         * @param[in] domainDescriptor describing the domain of the functional
         */
        explicit L1Norm(const DataDescriptor& domainDescriptor);

        /// make copy constructor deletion explicit
        L1Norm(const L1Norm<data_t>&) = delete;

        /// default destructor
        ~L1Norm() override = default;

    protected:
        /// the evaluation of the l1 norm
        data_t evaluateImpl(const DataContainer<data_t>& Rx) override;

        /// the computation of the gradient (in place)
        void getGradientImpl(const DataContainer<data_t>& Rx, DataContainer<data_t>& out) override;

        /// the computation of the Hessian
        LinearOperator<data_t> getHessianImpl(const DataContainer<data_t>& Rx) override;

        /// implement the polymorphic clone operation
        L1Norm<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Functional<data_t>& other) const override;
    };

} // namespace elsa
