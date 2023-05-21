#include "LB.h"
#include "DataContainer.h"
#include "LinearOperator.h"
#include "TypeCasts.hpp"
#include "Logger.h"

#include "spdlog/stopwatch.h"
#include "PowerIterations.h"

namespace elsa
{
    template <typename data_t>
    LB<data_t>::LB(const LinearOperator<data_t>& A, const DataContainer<data_t>& b,
                   ProximalOperator<data_t> prox, data_t mu, std::optional<data_t> beta,
                   data_t epsilon)
        : A_(A.clone()), b_(b), prox_(prox), mu_(mu), epsilon_(epsilon)
    {
        if (!beta.has_value()) {

            beta_ = data_t{0.45} / powerIterations(adjoint(*A_) * (*A_));
            Logger::get("LinearizedBregman")->info("Step length is chosen to be: {:8.5}", beta_);

        } else {
            beta_ = *beta;
        }
    }

    template <typename data_t>
    auto LB<data_t>::solve(index_t iterations, std::optional<DataContainer<data_t>> x0)
        -> DataContainer<data_t>
    {
        spdlog::stopwatch iter_time;

        auto v = DataContainer<data_t>(A_->getDomainDescriptor());
        auto x = DataContainer<data_t>(A_->getDomainDescriptor());

        if (x0.has_value()) {
            x = *x0;
        } else {
            x = 0;
        }

        auto A_transpose_b_ = A_->applyAdjoint(b_);
        auto prev_x = DataContainer<data_t>(A_->getDomainDescriptor());

        for (int i = 0; i < iterations; ++i) {

            auto Ax = A_->apply(x);
            auto A_transpose_Ax = A_->applyAdjoint(Ax);
            v += (A_transpose_b_ - A_transpose_Ax);
            x = beta_ * prox_.apply(v, mu_);

            auto delta = (Ax - b_).squaredL2Norm() / b_.squaredL2Norm();
            if (delta <= epsilon_) {
                Logger::get("LinearizedBregman")
                    ->info("SUCCESS: Reached convergence at {}/{} iteration", i + 1, iterations);
                return x;
            }

            Logger::get("LinearizedBregman")
                ->info("|iter: {:>6} | x: {:>12} | v: {:>12} | delta: {:>12} | time: {:>8.3} |", i,
                       x.squaredL2Norm(), v.squaredL2Norm(), delta, iter_time);
        }

        Logger::get("LinearizedBregman")
            ->warn("Failed to reach convergence at {} iterations", iterations);

        return x;
    }

    template <typename data_t>
    auto LB<data_t>::cloneImpl() const -> LB<data_t>*
    {
        return new LB<data_t>(*A_, b_, prox_, mu_, beta_, epsilon_);
    }

    template <typename data_t>
    auto LB<data_t>::isEqual(const Solver<data_t>& other) const -> bool
    {
        auto otherPGD = downcast_safe<LB>(&other);
        if (!otherPGD)
            return false;

        Logger::get("LinearizedBregman")->info("beta: {}, {}", beta_, otherPGD->beta_);
        if (std::abs(beta_ - otherPGD->beta_) > 1e-5)
            return false;

        Logger::get("LinearizedBregman")->info("mu: {}, {}", mu_, otherPGD->mu_);
        if (mu_ != otherPGD->mu_)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class LB<float>;
    template class LB<double>;
} // namespace elsa
