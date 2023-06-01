#include "APGD.h"
#include "DataContainer.h"
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
    APGD<data_t>::APGD(const LinearOperator<data_t>& A, const DataContainer<data_t>& b,
                       ProximalOperator<data_t> prox, std::optional<data_t> mu, data_t epsilon)
        : A_(A.clone()), b_(b), prox_(prox), mu_(0), epsilon_(epsilon)
    {
        if (!mu.has_value()) {
            Logger::get("APGD")->info("Computing Lipschitz constant to compute step size");
            spdlog::stopwatch time;

            // FISTA converges if \f$\mu \in (0, \frac{2}{L})\f$, where \f$L\f$
            // is the Lipschitz constant. A value just below the upper limit is chosen by default
            mu_ = data_t{0.45} / powerIterations(adjoint(*A_) * (*A_));
            Logger::get("APGD")->info("Step length is chosen to be: {:8.5}, (it took {}s)", mu_,
                                      time);
        } else {
            mu_ = *mu;
        }
    }

    template <typename data_t>
    auto APGD<data_t>::solve(index_t iterations, std::optional<DataContainer<data_t>> x0)
        -> DataContainer<data_t>
    {
        spdlog::stopwatch aggregate_time;

        auto x = extract_or(x0, A_->getDomainDescriptor());

        auto xPrev = x;
        auto y = x;
        auto z = x;
        data_t tPrev = 1;

        auto Atb = A_->applyAdjoint(b_);
        auto Ay = empty<data_t>(A_->getRangeDescriptor());
        auto grad = empty<data_t>(A_->getDomainDescriptor());

        // Compute gradient as A^T(A(x) - A^T(b)) memory efficient
        auto gradient = [&](const DataContainer<data_t>& x) {
            A_->apply(x, Ay);
            A_->applyAdjoint(Ay, grad);
            grad -= Atb;
        };

        Logger::get("APGD")->info("|*{:^6}*|*{:*^12}*|*{:*^8}*|*{:^8}*|", "iter", "gradient",
                                  "time", "elapsed");

        // Compute gradient here
        gradient(y);

        for (index_t iter = 0; iter < iterations; ++iter) {
            spdlog::stopwatch iter_time;

            // z = y - mu_ * grad
            lincomb(1, y, -mu_, grad, z);

            // x_{k+1} = prox_{mu * g}(y - mu * grad)
            x = prox_.apply(z, mu_);

            // t_{k+1} = \frac{\sqrt{1 + 4t_k^2} + 1}{2}
            data_t t = (1 + std::sqrt(1 + 4 * tPrev * tPrev)) / 2;

            // y_{k+1} = x_k + \frac{t_{k-1} - 1}{t_k}(x_k - x_{k-1})
            lincomb(1, x, (tPrev - 1) / t, x - xPrev, y); // 1 temporary

            xPrev = x;
            tPrev = t;

            if (grad.squaredL2Norm() <= epsilon_) {
                Logger::get("APGD")->info("SUCCESS: Reached convergence at {}/{} iteration",
                                          iter + 1, iterations);
                return x;
            }

            // Update gradient as a last step
            gradient(y);

            Logger::get("APGD")->info("| {:>6} | {:>12} | {:>8.3} | {:>8.3}s |", iter,
                                      grad.squaredL2Norm(), iter_time, aggregate_time);
        }

        Logger::get("APGD")->warn("Failed to reach convergence at {} iterations", iterations);

        return x;
    }

    template <typename data_t>
    auto APGD<data_t>::cloneImpl() const -> APGD<data_t>*
    {
        return new APGD<data_t>(*A_, b_, prox_, mu_, epsilon_);
    }

    template <typename data_t>
    auto APGD<data_t>::isEqual(const Solver<data_t>& other) const -> bool
    {
        auto otherAPGD = downcast_safe<APGD>(&other);
        if (!otherAPGD)
            return false;

        if (mu_ != otherAPGD->mu_)
            return false;

        if (epsilon_ != otherAPGD->epsilon_)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class APGD<float>;
    template class APGD<double>;
} // namespace elsa
