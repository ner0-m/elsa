#pragma once

#include <memory>

#include "DataContainer.h"
#include "LinearOperator.h"
#include "Solver.h"
#include "WLSProblem.h"
#include "TikhonovProblem.h"
#include "elsaDefines.h"

namespace elsa
{
    /// @brief Conjugate Gradient for Least Squares Problems
    ///
    /// CGLS minimizes:
    /// \f$ \frac{1}{2} || Ax - b ||_2^2 + \eps^2 || x || \f$
    /// where \f$A\f$ is an operator, it does not need to be square, symmetric, or positive
    /// definite. \f$b\f$ is the measured quantity, and \f$\eps\f$ a dampening factors.
    ///
    /// If the dampening factor \f$\eps\f$ is non zero, the problem solves a Tikhonov
    /// problem.
    ///
    /// CGLS is equivalent to apply CG to the normal equations \f$A^TAx = A^Tb\f$.
    /// However, it doesn not need to actually form the normal equation. Primarly,
    /// this improves the runtime performance, and it further improves the stability
    /// of the algorithm.
    ///
    /// References:
    /// - https://web.stanford.edu/group/SOL/software/cgls/
    ///
    /// @author David Frank
    template <typename data_t = real_t>
    class CGLS : public Solver<data_t>
    {
    public:
        /// Scalar alias
        using Scalar = typename Solver<data_t>::Scalar;

        /// @brief Construct the necessary form of CGLS using some linear operator and
        /// the measured data.
        ///
        /// @param A linear operator for the problem
        /// @param b the measured data
        /// @param eps the dampening factor
        /// @param tol stopping tolerance
        CGLS(const LinearOperator<data_t>& A, const DataContainer<data_t>& b,
             SelfType_t<data_t> eps = 0, SelfType_t<data_t> tol = 1e-4);

        /// @brief Construct CGLS from a least squares problem
        ///
        /// @param wls The least squares problem to construct the CGLS from
        /// @param tol stopping tolerance
        explicit CGLS(const WLSProblem<data_t>& wls, SelfType_t<data_t> tol = 1e-4);

        /// @brief Construct CGLS from a tikhonov problem
        ///
        /// @param tikhonov The Tikhonov problem to construct the CGLS from
        /// @param tol stopping tolerance
        explicit CGLS(const TikhonovProblem<data_t>& tikhonov, SelfType_t<data_t> tol = 1e-4);

        /// make copy constructor deletion explicit
        CGLS(const CGLS<data_t>&) = delete;

        /// default destructor
        ~CGLS() override = default;

        /**
         * @brief Solve the optimization problem, i.e. apply iterations number of iterations of CGLS
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

        data_t damp_{0};

        data_t tol_{0.0001};

        /// implement the polymorphic clone operation
        CGLS<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Solver<data_t>& other) const override;
    };
} // namespace elsa
