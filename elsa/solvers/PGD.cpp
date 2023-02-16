#include "PGD.h"
#include "LinearOperator.h"
#include "LinearResidual.h"
#include "ProximalL1.h"
#include "TypeCasts.hpp"
#include "Logger.h"
#include "PowerIterations.h"

#include "spdlog/stopwatch.h"

namespace elsa
{
    template <typename data_t>
    PGD<data_t>::PGD(const LinearOperator<data_t>& A, const DataContainer<data_t>& b,
                     ProximalOperator<data_t> prox, geometry::Threshold<data_t> mu, data_t epsilon)
        : A_(A.clone()), b_(b), prox_(prox), lambda_(1), mu_(data_t{mu}), epsilon_(epsilon)
    {
    }

    template <typename data_t>
    PGD<data_t>::PGD(const LinearOperator<data_t>& A, const DataContainer<data_t>& b,
                     ProximalOperator<data_t> prox, data_t epsilon)
        : A_(A.clone()), b_(b), prox_(prox), lambda_(1), epsilon_(epsilon)
    {
    }

    template <typename data_t>
    auto PGD<data_t>::solve(index_t iterations, std::optional<DataContainer<data_t>> x0)
        -> DataContainer<data_t>
    {
        spdlog::stopwatch aggregate_time;
        Logger::get("PGD")->info("Start preparations...");

        auto x = DataContainer<data_t>(A_->getDomainDescriptor());
        if (x0.has_value()) {
            x = *x0;
        } else {
            x = 0;
        }

        DataContainer<data_t> Atb = A_->applyAdjoint(b_);
        DataContainer<data_t> gradient = A_->applyAdjoint(A_->apply(x)) - Atb;

        if (!mu_.isInitialized()) {
            // ISTA converges if \f$\mu \in (0, \frac{2}{L})\f$, where \f$L\f$
            // is the Lipschitz constant. A value just below the upper limit is chosen by default
            mu_ = data_t{0.45} / powerIterations(adjoint(*A_) * (*A_));
        }

        Logger::get("PGD")->info("mu: {}", *mu_);

        Logger::get("PGD")->info("Preparations done, tooke {}s", aggregate_time);
        Logger::get("PGD")->info("{:^6}|{:*^16}|{:*^8}|{:*^8}|", "iter", "gradient", "time",
                                 "elapsed");

        auto deltaZero = gradient.squaredL2Norm();
        for (index_t iter = 0; iter < iterations; ++iter) {
            spdlog::stopwatch iter_time;

            gradient = A_->applyAdjoint(A_->apply(x)) - Atb;

            x = prox_.apply(x - *mu_ * gradient, geometry::Threshold{*mu_ * lambda_});

            Logger::get("PGD")->info("{:>5} |{:>15} | {:>6.3} |{:>6.3}s |", iter,
                                     gradient.squaredL2Norm(), iter_time, aggregate_time);
            if (gradient.squaredL2Norm() <= epsilon_ * epsilon_ * deltaZero) {
                Logger::get("PGD")->info("SUCCESS: Reached convergence at {}/{} iteration",
                                         iter + 1, iterations);
                return x;
            }
        }

        Logger::get("PGD")->warn("Failed to reach convergence at {} iterations", iterations);

        return x;
    }

    template <typename data_t>
    auto PGD<data_t>::cloneImpl() const -> PGD<data_t>*
    {
        if (mu_.isInitialized()) {
            return new PGD(*A_, b_, prox_, geometry::Threshold<data_t>{*mu_}, epsilon_);
        } else {
            return new PGD(*A_, b_, prox_, epsilon_);
        }
    }

    template <typename data_t>
    auto PGD<data_t>::isEqual(const Solver<data_t>& other) const -> bool
    {
        auto otherPGD = downcast_safe<PGD>(&other);
        if (!otherPGD)
            return false;

        if (mu_.isInitialized() != otherPGD->mu_.isInitialized())
            return false;

        if (mu_ != otherPGD->mu_)
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
