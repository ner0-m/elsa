#pragma once

#include <optional>

#include "DataContainer.h"
#include "Solver.h"
#include "Scaling.h"
#include "MaybeUninitialized.hpp"

namespace elsa
{
    /// @brief Default constraint for Landweber Iterations, i.e. no constraint
    struct IdentityProjection {
        template <class data_t>
        void operator()([[maybe_unused]] DataContainer<data_t>& x) const
        {
        }
    };

    /// @brief Constraint functor, which constraints a `DataContainer` such that,
    /// it for all values \f$x_i\f$ it holds: \f$0 < x_i < 1\f$
    struct BoxConstraint {
        template <class data_t>
        void operator()(DataContainer<data_t>& x) const
        {
            x = clip(x, static_cast<data_t>(0), static_cast<data_t>(1));
        }
    };

    /// @brief Constraint functor, which constraints a `DataContainer` such that,
    /// it for all values \f$x_i\f$ it holds: \f$0 < x_i\f$
    struct NonNegativeConstraint {
        template <class data_t>
        void operator()(DataContainer<data_t>& x) const
        {
            // TODO: This could be optimized, but for now this works
            auto maxVal = x.maxElement();
            x = clip(x, static_cast<data_t>(0), maxVal);
        }
    };

    /**
     * @brief Base class for Landweber iteration. These are a group of algorithms solving ill-posed
     * linear inverse problem.
     *
     * The algorithms solve the equation of the form \f$A x = b\f$, by iteratively solving
     * \f$ \argmin_x \frac{1}{2} \| Ax - b \|_{2}^2 \f$. The general update rule for Landwehr
     * iterations is:
     *
     * - \f$ x_{k+1} =  x_{k} + \lambda T A^T M (A(x_{k}) - b)\f$
     *
     * Here, \f$k\f$ is the number of the current iteration, \f$b\f$ the measured data (i.e.
     * sinogram), \f$A\f$ an operator (i.e. the Radon transform). \f$T\f$ and \f$M\f$ are method
     * specific, the exact choice of these determine the exact method. Among others:
     *
     * - Given \f$ T = M = I \f$ then the algorithm is refered to as the classical Landweber
     *   iteration
     * - Given \f$ T = I\f$ \f$ M = \frac{1}{m} \text{diag}(\frac{1}{ \| a_i \|^2_2 }) \f$ (where
     *   \f$ \| a_i \|^2_2 \f$ is the \f$\ell^2\f$-norm of the \f$i\f$-th row of the operator)
     *   referred to as Cimmino's method
     * - Given \f$ T = \text{diag}(\text{row sum})^{-1}\f$ \f$ M = \text{diag}(\text{col sum})^1\f$
     *   it is referred to as SART
     *
     * @author David Frank
     */
    template <typename data_t = real_t>
    class LandweberIteration : public Solver<data_t>
    {
    public:
        /// Scalar alias
        using Scalar = typename Solver<data_t>::Scalar;

        /**
         * @brief Constructor for Landweber type solver, accepting an operator, a measurement vector
         * and a step size#
         *
         * @param[in] A linear operator to solve the problem with
         * @param[in] b measurment vector of the problem
         * @param[in] stepSize the fixed step size to be used while solving
         */
        LandweberIteration(const LinearOperator<data_t>& A, const DataContainer<data_t>& b,
                           data_t stepSize);

        /**
         * @brief Constructor for Landweber type solver, accepting an operator and a measurement
         * vector
         *
         * @param[in] A linear operator to solve the problem with
         * @param[in] b measurment vector of the problem
         */
        LandweberIteration(const LinearOperator<data_t>& A, const DataContainer<data_t>& b);

        /// make copy constructor deletion explicit
        LandweberIteration(const LandweberIteration<data_t>&) = delete;

        /// default destructor
        ~LandweberIteration() override = default;

        DataContainer<data_t>
            solve(index_t iterations,
                  std::optional<DataContainer<data_t>> x0 = std::nullopt) override;

        void setProjection(const std::function<void(DataContainer<data_t>&)> projection);

    protected:
        /// Setup the \f$T * A^T * M\f& operator, implemented by the base classes to allow for
        /// different types of solvers
        virtual std::unique_ptr<LinearOperator<data_t>>
            setupOperators(const LinearOperator<data_t>& A) const = 0;

        /// the linear operator
        std::unique_ptr<LinearOperator<data_t>> A_;

        /// the measurement data
        DataContainer<data_t> b_;

        std::function<void(DataContainer<data_t>&)> projection_ = IdentityProjection{};

        /// The composition of T, A^T and M to apply in each update
        std::unique_ptr<LinearOperator<data_t>> tam_{};

        /// the step size
        MaybeUninitialized<data_t> stepSize_;

        /// implement the polymorphic comparison operation
        bool isEqual(const Solver<data_t>& other) const override;
    };
} // namespace elsa
