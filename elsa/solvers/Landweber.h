#pragma once

#include "LandweberIteration.h"
#include "elsaDefines.h"

namespace elsa
{
    /**
     * @brief Implementation of the Landweber iterations with both \f$ T \f$ and \f$ M \f$ being the
     * identity matrix. This reduces the update rule to:
     *
     * - \f$ x_{k+1} =  x_{k} + \lambda A^T (A(x_{k}) - b)\f$
     *
     * This is basically a special case of the gradient descent.
     *
     * @author David Frank
     * @see LandweberIteration
     */
    template <typename data_t = real_t>
    class Landweber : public LandweberIteration<data_t>
    {
    public:
        /// Scalar alias
        using Scalar = typename LandweberIteration<data_t>::Scalar;

        /**
         * @brief Constructor for classical Landweber, accepting an operator, a measurement vector
         * and a step size#
         *
         * @param[in] A linear operator to solve the problem with
         * @param[in] b measurment vector of the problem
         * @param[in] stepSize the fixed step size to be used while solving
         */
        Landweber(const LinearOperator<data_t>& A, const DataContainer<data_t>& b,
                  SelfType_t<data_t> stepSize);

        /**
         * @brief Constructor for classical Landweber, accepting an operator and a measurement
         * vector
         *
         * @param[in] A linear operator to solve the problem with
         * @param[in] b measurment vector of the problem
         */
        Landweber(const LinearOperator<data_t>& A, const DataContainer<data_t>& b);

        /**
         * @brief Constructor for Landweber, accepting a problem and a fixed step size
         *
         * @param[in] problem the problem that is supposed to be solved
         * @param[in] stepSize the fixed step size to be used while solving
         */
        Landweber(const WLSProblem<data_t>& problem, data_t stepSize);

        /**
         * @brief Constructor for Landweber, accepting a problem
         *
         * @param[in] problem the problem that is supposed to be solved
         */
        explicit Landweber(const WLSProblem<data_t>& problem);

        /// make copy constructor deletion explicit
        Landweber(const Landweber<data_t>&) = delete;

        /// default destructor
        ~Landweber() override = default;

    protected:
        std::unique_ptr<LinearOperator<data_t>>
            setupOperators(const LinearOperator<data_t>& A) const override;

    private:
        /// implement the polymorphic clone operation
        Landweber<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Solver<data_t>& other) const override;
    };
} // namespace elsa
