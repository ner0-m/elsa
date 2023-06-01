#include "PGD.h"
#include "DataContainer.h"
#include "LinearOperator.h"
#include "LinearResidual.h"
#include "ProximalL1.h"
#include "Solver.h"
#include "TypeCasts.hpp"
#include "Logger.h"
#include "PowerIterations.h"

#include "spdlog/stopwatch.h"

namespace elsa
{
    template <typename data_t>
    PGD<data_t>::PGD(const LinearOperator<data_t>& A, const DataContainer<data_t>& b,
                     ProximalOperator<data_t> prox, std::optional<data_t> mu, data_t epsilon)
        : A_(A.clone()), b_(b), prox_(prox), mu_(0), epsilon_(epsilon)
    {
        if (!mu.has_value()) {
            Logger::get("PGD")->info("Computing Lipschitz constant to compute step size");
            spdlog::stopwatch time;

            // ISTA converges if \f$\mu \in (0, \frac{2}{L})\f$, where \f$L\f$
            // is the Lipschitz constant. A value just below the upper limit is chosen by default
            mu_ = data_t{0.45} / powerIterations(adjoint(*A_) * (*A_));
            Logger::get("PGD")->info("Step length is chosen to be: {:8.5}, (it took {}s)", mu_,
                                     time);
        } else {
            mu_ = *mu;
        }
    }

    template <typename data_t>
    auto PGD<data_t>::solve(index_t iterations, std::optional<DataContainer<data_t>> x0)
        -> DataContainer<data_t>
    {
        spdlog::stopwatch aggregate_time;

        auto x = extract_or(x0, A_->getDomainDescriptor());

        auto Atb = A_->applyAdjoint(b_);
        auto Ay = empty<data_t>(A_->getRangeDescriptor());
        auto grad = empty<data_t>(A_->getDomainDescriptor());

        // Compute gradient as A^T(A(x) - A^T(b)) memory efficient
        auto gradient = [&](const DataContainer<data_t>& x) {
            A_->apply(x, Ay);
            A_->applyAdjoint(Ay, grad);
            grad -= Atb;
        };

        Logger::get("PGD")->info("Preparations done, tooke {}s", aggregate_time);
        Logger::get("PGD")->info("|*{:^6}*|*{:*^12}*|*{:*^8}*|*{:^8}*|", "iter", "gradient", "time",
                                 "elapsed");

        auto y = emptylike(x);

        for (index_t iter = 0; iter < iterations; ++iter) {
            spdlog::stopwatch iter_time;

            gradient(x);

            // y = x - mu_ * grad
            lincomb(1, x, -mu_, grad, y);

            // apply proximal
            x = prox_.apply(y, mu_);

            if (grad.squaredL2Norm() <= epsilon_) {
                Logger::get("PGD")->info("SUCCESS: Reached convergence at {}/{} iteration",
                                         iter + 1, iterations);
                return x;
            }

            Logger::get("PGD")->info("| {:>6} | {:>12} | {:>8.3} | {:>8.3}s |", iter,
                                     grad.squaredL2Norm(), iter_time, aggregate_time);
        }

        Logger::get("PGD")->warn("Failed to reach convergence at {} iterations", iterations);

        return x;
    }

    template <typename data_t>
    auto PGD<data_t>::cloneImpl() const -> PGD<data_t>*
    {
        return new PGD<data_t>(*A_, b_, prox_, mu_, epsilon_);
    }

    template <typename data_t>
    auto PGD<data_t>::isEqual(const Solver<data_t>& other) const -> bool
    {
        auto otherPGD = downcast_safe<PGD>(&other);
        if (!otherPGD)
            return false;

        Logger::get("PGD")->info("mu: {}, {}", mu_, otherPGD->mu_);
        if (std::abs(mu_ - otherPGD->mu_) > 1e-5)
            return false;

        if (epsilon_ != otherPGD->epsilon_)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class PGD<float>;
    template class PGD<double>;
} // namespace elsa
