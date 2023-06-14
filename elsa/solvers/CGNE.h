#pragma once

#include <memory>

#include "DataContainer.h"
#include "LinearOperator.h"
#include "Solver.h"
#include "elsaDefines.h"

namespace elsa
{
    /// @brief Conjugate Gradient via the Normal equation
    ///
    /// CG solves the system of equations:
    /// \[
    /// A x = b
    /// \]
    /// where \f$A\f$ is symmetric positive definite operator. \f$b\f$ is the measured quantity.
    ///
    /// In our implementation, we always assume \f$A\f$ is non-symmetric and not positive
    /// definite. Hence, we compute the solution to the normal equation
    /// \[
    /// A^T A x = A^t b
    /// \]
    ///
    /// References:
    /// - An Introduction to the Conjugate Gradient Method Without the Agonizing Pain, by Shewchuk
    ///
    /// @author David Frank
    template <typename data_t = real_t>
    class CGNE : public Solver<data_t>
    {
    public:
        /// Scalar alias
        using Scalar = typename Solver<data_t>::Scalar;

        /// @brief Construct the necessary form of CGNE using some linear operator and
        /// the measured data.
        ///
        /// @param A linear operator for the problem
        /// @param b the measured data
        /// @param tol stopping tolerance
        CGNE(const LinearOperator<data_t>& A, const DataContainer<data_t>& b,
             SelfType_t<data_t> tol = 1e-4);

        /// make copy constructor deletion explicit
        CGNE(const CGNE<data_t>&) = delete;

        /// default destructor
        ~CGNE() override = default;

        /**
         * @brief Solve the optimization problem, i.e. apply iterations number of iterations of CGNE
         *
         * @param[in] iterations number of iterations to execute
         * @param[in] x0 optional initial solution, initial solution set to zero if not present
         *
         * @returns the approximated solution
         */
        DataContainer<data_t>
            solve(index_t iterations,
                  std::optional<DataContainer<data_t>> x0 = std::nullopt) override;

    private:
        std::unique_ptr<LinearOperator<data_t>> A_;

        DataContainer<data_t> b_;

        data_t tol_{0.0001};

        /// implement the polymorphic clone operation
        CGNE<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Solver<data_t>& other) const override;
    };
} // namespace elsa
