#include "CGNE.h"
#include "DataContainer.h"
#include "Error.h"
#include "L2NormPow2.h"
#include "LinearOperator.h"
#include "LinearResidual.h"
#include "Solver.h"
#include "TypeCasts.hpp"
#include "spdlog/stopwatch.h"
#include "Logger.h"

namespace elsa
{
    template <class data_t>
    CGNE<data_t>::CGNE(const LinearOperator<data_t>& A, const DataContainer<data_t>& b,
                       SelfType_t<data_t> tol)
        : A_(A.clone()), b_(b), tol_(tol)
    {
    }

    template <class data_t>
    DataContainer<data_t> CGNE<data_t>::solve(index_t iterations,
                                              std::optional<DataContainer<data_t>> x0)
    {
        spdlog::stopwatch aggregate_time;

        auto x = DataContainer<data_t>(A_->getDomainDescriptor());

        if (x0.has_value()) {
            x = *x0;
        } else {
            x = 0;
        }

        // setup A^T * A, and A^T * b
        auto A = adjoint(*A_) * (*A_);
        auto b = A_->applyAdjoint(b_);

        // Residual b - A(x)
        auto r = A.apply(x);
        r *= data_t{-1};
        r += b;

        auto c = r;

        // Squared Norm of residual
        auto kold = r.squaredL2Norm();

        auto Ac = DataContainer<data_t>(A.getRangeDescriptor());

        Logger::get("CGNE")->info("{:^5} | {:^15} | {:^15} | {:^15} | {:^15} |", "Iters", "r", "c",
                                  "alpha", "beta");
        for (int iter = 0; iter < iterations; ++iter) {
            A.apply(c, Ac);
            auto cAc = c.dot(Ac);

            auto alpha = kold / cAc;

            // Update x and residual
            x += alpha * c;
            r -= alpha * Ac;

            auto k = r.squaredL2Norm();
            auto beta = k / kold;

            // c = r + beta * c
            c = r;
            c += beta * c;

            // store k for next iteration
            kold = k;

            Logger::get("CGNE")->info("{:>5} | {:>15.10} | {:>15.10} | {:>15.10} | {:>15.10} |",
                                      iter, r.l2Norm(), c.l2Norm(), alpha, beta);
        }

        Logger::get("CGNE")->warn("Failed to reach convergence at {} iterations", iterations);

        return x;
    }
    template <class data_t>
    CGNE<data_t>* CGNE<data_t>::cloneImpl() const
    {
        return new CGNE(*A_, b_, tol_);
    }

    template <class data_t>
    bool CGNE<data_t>::isEqual(const Solver<data_t>& other) const
    {
        auto cgne = downcast_safe<CGNE>(&other);
        return cgne && *cgne->A_ == *A_ && cgne->b_ == b_;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class CGNE<float>;
    template class CGNE<double>;

} // namespace elsa
