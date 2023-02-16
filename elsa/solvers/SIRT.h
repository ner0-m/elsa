#pragma once

#include "LandweberIteration.h"
#include "elsaDefines.h"

namespace elsa
{
    /**
     * @brief Implementation of the Simultaneous Iterative Reconstruction
     * Technique (SIRT). For SIRT \f$ T = \text{diag}(\frac{1}{\text{row sum}})
     * = \text{diag}(\frac{1}{\sum_i A_{ij}})\f$ and \f$ M = \text{diag}(
     * \frac{i}{\text{col sum}}) = \text{diag}(\frac{1}{\sum_j A_{ij}})\f$.
     *
     * Outside of the Computed Tomography community, this algorithm is also often
     * know as Simultaneous Algebraic Reconstruction Technique (SART).
     *
     * @author David Frank
     * @see LandweberIteration
     */
    template <typename data_t = real_t>
    class SIRT : public LandweberIteration<data_t>
    {
    public:
        /// Scalar alias
        using Scalar = typename LandweberIteration<data_t>::Scalar;

        /**
         * @brief Constructor for SIRT, accepting an operator, a measurement vector
         * and a step size#
         *
         * @param[in] A linear operator to solve the problem with
         * @param[in] b measurment vector of the problem
         * @param[in] stepSize the fixed step size to be used while solving
         */
        SIRT(const LinearOperator<data_t>& A, const DataContainer<data_t>& b,
             SelfType_t<data_t> stepSize);

        /**
         * @brief Constructor for SIRT, accepting an operator and a measurement
         * vector
         *
         * @param[in] A linear operator to solve the problem with
         * @param[in] b measurment vector of the problem
         */
        SIRT(const LinearOperator<data_t>& A, const DataContainer<data_t>& b);

        /// make copy constructor deletion explicit
        SIRT(const SIRT<data_t>&) = delete;

        /// default destructor
        ~SIRT() override = default;

    protected:
        std::unique_ptr<LinearOperator<data_t>>
            setupOperators(const LinearOperator<data_t>& wls) const override;

    private:
        /// implement the polymorphic clone operation
        SIRT<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Solver<data_t>& other) const override;
    };
} // namespace elsa
