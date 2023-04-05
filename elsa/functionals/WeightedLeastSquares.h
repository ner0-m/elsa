#pragma once

#include "DataDescriptor.h"
#include "Functional.h"
#include "DataContainer.h"
#include "LinearOperator.h"
#include "Scaling.h"

namespace elsa
{
    /**
     * @brief The least squares functional / loss functional.
     *
     * The least squares loss is given by:
     * \[
     * \frac{1}{2} || A(x) - b ||_2^2
     * \]
     * i.e. the squared \f$\ell^2\f$ of the linear residual.
     *
     * @tparam data_t data type for the domain of the residual of the functional, defaulting to
     * real_t
     */
    template <typename data_t = real_t>
    class WeightedLeastSquares : public Functional<data_t>
    {
    public:
        /**
         * @brief Constructor the l2 norm (squared) functional with a LinearResidual
         *
         * @param[in] A LinearOperator to use in the residual
         * @param[in] b data to use in the linear residual
         */
        WeightedLeastSquares(const LinearOperator<data_t>& A, const DataContainer<data_t>& b,
                             const DataContainer<data_t>& weights);

        /// make copy constructor deletion explicit
        WeightedLeastSquares(const WeightedLeastSquares<data_t>&) = delete;

        /// default destructor
        ~WeightedLeastSquares() override = default;

        bool isDifferentiable() const override;

        const LinearOperator<data_t>& getOperator() const;

        const DataContainer<data_t>& getDataVector() const;

    protected:
        /// the evaluation of the l2 norm (squared)
        data_t evaluateImpl(const DataContainer<data_t>& Rx) override;

        /// the computation of the gradient (in place)
        void getGradientImpl(const DataContainer<data_t>& Rx, DataContainer<data_t>& out) override;

        /// the computation of the Hessian
        LinearOperator<data_t> getHessianImpl(const DataContainer<data_t>& Rx) override;

        /// implement the polymorphic clone operation
        WeightedLeastSquares<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Functional<data_t>& other) const override;

    private:
        std::unique_ptr<LinearOperator<data_t>> A_{};

        DataContainer<data_t> b_{};

        Scaling<data_t> W_;
    };

} // namespace elsa
