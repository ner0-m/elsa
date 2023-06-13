#pragma once

#include "DataContainer.h"
#include "Functional.h"

namespace elsa
{
    /**
     * @brief Class representing the l0 pseudo-norm functional.
     *
     * The l0 pseudo-norm functional evaluates to @f$ \sum_{i=1}^n 1_{x_{i} \neq 0} @f$ for @f$
     * x=(x_i)_{i=1}^n @f$. Please note that it is not differentiable, hence getGradient and
     * getHessian will throw exceptions.
     *
     * References:
     * * https://www.sciencedirect.com/bookseries/north-holland-mathematical-library/vol/38/suppl/C
     * * https://en.wikipedia.org/wiki/Lp_space#When_p_=_0
     *
     * @tparam data_t data type for the domain of the residual of the functional, defaulting to
     * real_t
     *
     * @author Andi Braimllari - initial code
     */
    template <typename data_t = real_t>
    class L0PseudoNorm : public Functional<data_t>
    {
    public:
        /**
         * @brief Constructor for the l0 pseudo-norm functional, mapping domain vector to a scalar
         * (without a residual)
         *
         * @param[in] domainDescriptor describing the domain of the functional
         */
        explicit L0PseudoNorm(const DataDescriptor& domainDescriptor);

        /// make copy constructor deletion explicit
        L0PseudoNorm(const L0PseudoNorm<data_t>&) = delete;

        /// default destructor
        ~L0PseudoNorm() override = default;

    protected:
        /// the evaluation of the l0 pseudo-norm
        auto evaluateImpl(const DataContainer<data_t>& Rx) -> data_t override;

        /// the computation of the gradient (in place)
        void getGradientImpl(const DataContainer<data_t>& Rx, DataContainer<data_t>&) override;

        /// the computation of the Hessian
        auto getHessianImpl(const DataContainer<data_t>& Rx) -> LinearOperator<data_t> override;

        /// implement the polymorphic clone operation
        auto cloneImpl() const -> L0PseudoNorm<data_t>* override;

        /// implement the polymorphic comparison operation
        auto isEqual(const Functional<data_t>& other) const -> bool override;
    };
} // namespace elsa
