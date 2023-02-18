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
                       ProximalOperator<data_t> prox, geometry::Threshold<data_t> mu,
                       data_t epsilon)
        : A_(A.clone()), b_(b), prox_(prox), mu_(data_t{mu}), epsilon_(epsilon)
    {
    }

    template <typename data_t>
    APGD<data_t>::APGD(const LinearOperator<data_t>& A, const DataContainer<data_t>& b,
                       ProximalOperator<data_t> prox, data_t epsilon)
        : A_(A.clone()), b_(b), prox_(prox), epsilon_(epsilon)
    {
    }

    template <typename data_t>
    APGD<data_t>::APGD(const LASSOProblem<data_t>& problem, geometry::Threshold<data_t> mu,
                       data_t epsilon)
        : Solver<data_t>(),
          A_(downcast<LinearResidual<data_t>>(problem.getDataTerm().getResidual())
                 .getOperator()
                 .clone()),
          b_(downcast<LinearResidual<data_t>>(problem.getDataTerm().getResidual()).getDataVector()),
          prox_(ProximalL1<data_t>()),
          lambda_(problem.getRegularizationTerms()[0].getWeight()),
          mu_{data_t(mu)},
          epsilon_{epsilon}
    {
    }

    template <typename data_t>
    APGD<data_t>::APGD(const Problem<data_t>& problem, geometry::Threshold<data_t> mu,
                       data_t epsilon)
        : APGD(LASSOProblem<data_t>(problem), mu, epsilon)
    {
    }

    template <typename data_t>
    APGD<data_t>::APGD(const Problem<data_t>& problem, data_t epsilon)
        : APGD<data_t>(LASSOProblem<data_t>(problem), epsilon)
    {
    }

    template <typename data_t>
    APGD<data_t>::APGD(const LASSOProblem<data_t>& problem, data_t epsilon)
        : Solver<data_t>(),
          A_(downcast<LinearResidual<data_t>>(problem.getDataTerm().getResidual())
                 .getOperator()
                 .clone()),
          b_(downcast<LinearResidual<data_t>>(problem.getDataTerm().getResidual()).getDataVector()),
          prox_(ProximalL1<data_t>()),
          lambda_(problem.getRegularizationTerms()[0].getWeight()),
          epsilon_{epsilon}
    {
    }

    template <typename data_t>
    auto APGD<data_t>::solve(index_t iterations, std::optional<DataContainer<data_t>> x0)
        -> DataContainer<data_t>
    {
        spdlog::stopwatch aggregate_time;
        Logger::get("APGD")->info("Start preparations...");

        auto x = DataContainer<data_t>(A_->getDomainDescriptor());
        if (x0.has_value()) {
            x = *x0;
        } else {
            x = 0;
        }

        DataContainer<data_t> xPrev = x;
        DataContainer<data_t> y = x;
        DataContainer<data_t> yPrev = x;
        data_t t;
        data_t tPrev = 1;

        if (!mu_.isInitialized()) {
            // FISTA converges if \f$\mu \in (0, \frac{2}{L})\f$, where \f$L\f$
            // is the Lipschitz constant. A value just below the upper limit is chosen by default
            mu_ = data_t{0.45} / powerIterations(adjoint(*A_) * (*A_));
        }

        DataContainer<data_t> Atb = A_->applyAdjoint(b_);
        DataContainer<data_t> gradient = A_->applyAdjoint(A_->apply(yPrev)) - Atb;

        Logger::get("APGD")->info("Preparations done, tooke {}s", aggregate_time);

        Logger::get("APGD")->info("{:^6}|{:*^16}|{:*^8}|{:*^8}|", "iter", "gradient", "time",
                                  "elapsed");

        auto deltaZero = gradient.squaredL2Norm();
        for (index_t iter = 0; iter < iterations; ++iter) {
            spdlog::stopwatch iter_time;

            gradient = A_->applyAdjoint(A_->apply(yPrev)) - Atb;
            x = prox_.apply(yPrev - *mu_ * gradient, geometry::Threshold{*mu_ * lambda_});

            t = (1 + std::sqrt(1 + 4 * tPrev * tPrev)) / 2;
            y = x + ((tPrev - 1) / t) * (x - xPrev);

            xPrev = x;
            yPrev = y;
            tPrev = t;

            Logger::get("APGD")->info("{:>5} |{:>15} | {:>6.3} |{:>6.3}s |", iter,
                                      gradient.squaredL2Norm(), iter_time, aggregate_time);

            if (gradient.squaredL2Norm() <= epsilon_ * epsilon_ * deltaZero) {
                Logger::get("APGD")->info("SUCCESS: Reached convergence at {}/{} iteration",
                                          iter + 1, iterations);
                return x;
            }
        }

        Logger::get("APGD")->warn("Failed to reach convergence at {} iterations", iterations);

        return x;
    }

    template <typename data_t>
    auto APGD<data_t>::cloneImpl() const -> APGD<data_t>*
    {
        if (mu_.isInitialized()) {
            return new APGD(*A_, b_, prox_, geometry::Threshold<data_t>{*mu_}, epsilon_);
        } else {
            return new APGD(*A_, b_, prox_, epsilon_);
        }
    }

    template <typename data_t>
    auto APGD<data_t>::isEqual(const Solver<data_t>& other) const -> bool
    {
        auto otherAPGD = downcast_safe<APGD>(&other);
        if (!otherAPGD)
            return false;

        if (mu_.isInitialized() != otherAPGD->mu_.isInitialized())
            return false;

        if (*mu_ != *otherAPGD->mu_)
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
