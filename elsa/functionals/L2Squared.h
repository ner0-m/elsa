#pragma once

#include "DataDescriptor.h"
#include "Functional.h"
#include "DataContainer.h"
#include "LinearOperator.h"

namespace elsa
{
    /**
     * @brief Class representing the squared l2 norm functional.
     *
     * The l2 norm (squared) functional evaluates to \f$ 0.5 * \sum_{i=1}^n x_i^2 \f$ for \f$
     * x=(x_i)_{i=1}^n \f$.
     *
     * @tparam data_t data type for the domain of the residual of the functional, defaulting to
     * real_t
     */
    template <typename data_t = real_t>
    class L2Squared : public Functional<data_t>
    {
    public:
        /**
         * @brief Constructor for the l2 norm (squared) functional, mapping domain vector to a
         * scalar (without a residual)
         *
         * @param[in] domainDescriptor describing the domain of the functional
         */
        explicit L2Squared(const DataDescriptor& domainDescriptor);

        /**
         * @brief Constructor the l2 norm (squared) functional with a LinearResidual
         *
         * @param[in] domainDescriptor describing the domain of the functional
         * @param[in] b data to use in the linear residual
         */
        L2Squared(const DataContainer<data_t>& b);

        /// make copy constructor deletion explicit
        L2Squared(const L2Squared<data_t>&) = delete;

        /// default destructor
        ~L2Squared() override = default;

        bool hasDataVector() const;

        const DataContainer<data_t>& getDataVector() const;

    protected:
        /// the evaluation of the l2 norm (squared)
        data_t evaluateImpl(const DataContainer<data_t>& Rx) override;

        /// the computation of the gradient (in place)
        void getGradientImpl(const DataContainer<data_t>& Rx, DataContainer<data_t>& out) override;

        /// the computation of the Hessian
        LinearOperator<data_t> getHessianImpl(const DataContainer<data_t>& Rx) override;

        /// implement the polymorphic clone operation
        L2Squared<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Functional<data_t>& other) const override;

    private:
        std::unique_ptr<LinearOperator<data_t>> A_{};

        std::optional<DataContainer<data_t>> b_{};
    };

} // namespace elsa
