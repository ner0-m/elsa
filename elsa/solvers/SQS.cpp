#include "SQS.h"
#include "Identity.h"
#include "Scaling.h"
#include "Logger.h"
#include "TypeCasts.hpp"

namespace elsa
{
    template <typename data_t>
    SQS<data_t>::SQS(const L2NormPow2<data_t>& problem,
                     std::vector<std::unique_ptr<L2NormPow2<data_t>>>&& subsets,
                     bool momentumAcceleration, data_t epsilon)
        : Solver<data_t>(),
          fullProblem_(downcast<L2NormPow2<data_t>>(problem.clone())),
          subsets_(std::move(subsets)),
          epsilon_{epsilon},
          momentumAcceleration_{momentumAcceleration},
          subsetMode_(!subsets.empty())

    {
        Logger::get("SQS")->info("SQS running in ordered subset mode");
    }

    template <typename data_t>
    SQS<data_t>::SQS(const L2NormPow2<data_t>& problem,
                     std::vector<std::unique_ptr<L2NormPow2<data_t>>>&& subsets,
                     const LinearOperator<data_t>& preconditioner, bool momentumAcceleration,
                     data_t epsilon)
        : Solver<data_t>(),
          fullProblem_(downcast<L2NormPow2<data_t>>(problem.clone())),
          subsets_(std::move(subsets)),
          preconditioner_{preconditioner.clone()},
          epsilon_{epsilon},
          momentumAcceleration_{momentumAcceleration},
          subsetMode_(!subsets.empty())
    {
        Logger::get("SQS")->info("SQS running in ordered subset mode");

        // check that preconditioner is compatible with problem
        if (preconditioner_->getDomainDescriptor().getNumberOfCoefficients()
                != fullProblem_->getDomainDescriptor().getNumberOfCoefficients()
            || preconditioner_->getRangeDescriptor().getNumberOfCoefficients()
                   != fullProblem_->getDomainDescriptor().getNumberOfCoefficients()) {
            throw InvalidArgumentError("SQS: incorrect size of preconditioner");
        }
    }

    template <typename data_t>
    SQS<data_t>::SQS(const L2NormPow2<data_t>& problem, bool momentumAcceleration, data_t epsilon)
        : Solver<data_t>(),
          fullProblem_(downcast<L2NormPow2<data_t>>(problem.clone())),
          epsilon_{epsilon},
          momentumAcceleration_{momentumAcceleration},
          subsetMode_(false)

    {
        Logger::get("SQS")->info("SQS running in normal mode");
    }

    template <typename data_t>
    SQS<data_t>::SQS(const L2NormPow2<data_t>& problem,
                     const LinearOperator<data_t>& preconditioner, bool momentumAcceleration,
                     data_t epsilon)
        : Solver<data_t>(),
          fullProblem_(downcast<L2NormPow2<data_t>>(problem.clone())),
          preconditioner_{preconditioner.clone()},
          epsilon_{epsilon},
          momentumAcceleration_{momentumAcceleration},
          subsetMode_(false)
    {
        Logger::get("SQS")->info("SQS running in normal mode");

        // check that preconditioner is compatible with problem
        if (preconditioner_->getDomainDescriptor().getNumberOfCoefficients()
                != fullProblem_->getDomainDescriptor().getNumberOfCoefficients()
            || preconditioner_->getRangeDescriptor().getNumberOfCoefficients()
                   != fullProblem_->getDomainDescriptor().getNumberOfCoefficients()) {
            throw InvalidArgumentError("SQS: incorrect size of preconditioner");
        }
    }

    template <typename data_t>
    DataContainer<data_t> SQS<data_t>::solve(index_t iterations,
                                             std::optional<DataContainer<data_t>> x0)
    {
        auto& solutionDesc = fullProblem_->getDomainDescriptor();
        auto x = DataContainer<data_t>(solutionDesc);
        if (x0.has_value()) {
            x = *x0;
        } else {
            x = 0;
        }

        auto convergenceThreshold =
            fullProblem_->getGradient(x).squaredL2Norm() * epsilon_ * epsilon_;

        auto hessian = fullProblem_->getHessian(x);

        auto ones = DataContainer<data_t>(solutionDesc);
        ones = 1;
        auto diagVector = hessian.apply(ones);
        diagVector = static_cast<data_t>(1.0) / diagVector;
        auto diag = Scaling<data_t>(hessian.getDomainDescriptor(), diagVector);

        data_t prevT = 1;
        data_t t = 1;
        data_t nextT = 0;

        auto& z = x;
        // z = 0;
        DataContainer<data_t> prevX = x;
        DataContainer<data_t> gradient(solutionDesc);

        index_t nSubsets = subsetMode_ ? subsets_.size() : 1;

        for (index_t i = 0; i < iterations; i++) {
            Logger::get("SQS")->info("iteration {} of {}", i + 1, iterations);

            for (index_t m = 0; m < nSubsets; m++) {
                if (subsetMode_) {
                    subsets_[m]->getGradient(x, gradient);
                } else {
                    fullProblem_->getGradient(x, gradient);
                }

                if (preconditioner_) {
                    preconditioner_->apply(gradient, gradient);
                }

                // TODO: element wise relu
                if (momentumAcceleration_) {
                    nextT = as<data_t>(1)
                            + std::sqrt(as<data_t>(1) + as<data_t>(4) * t * t) / as<data_t>(2);

                    x = z - nSubsets * diag.apply(gradient);
                    z = x + prevT / nextT * (x - prevX);
                } else {
                    z = z - nSubsets * diag.apply(gradient);
                }

                // if the gradient is too small we stop
                if (gradient.squaredL2Norm() <= convergenceThreshold) {
                    if (!subsetMode_
                        || fullProblem_->getGradient(x).squaredL2Norm() <= convergenceThreshold) {
                        Logger::get("SQS")->info("SUCCESS: Reached convergence at {}/{} iteration",
                                                 i + 1, iterations);

                        // TODO: make return more sane
                        if (momentumAcceleration_) {
                            z = x;
                        }
                        return x;
                    }
                }

                if (momentumAcceleration_) {
                    prevT = t;
                    t = nextT;
                    prevX = x;
                }
            }
        }

        Logger::get("SQS")->warn("Failed to reach convergence at {} iterations", iterations);

        // TODO: make return more sane
        if (momentumAcceleration_) {
            z = x;
        }
        return x;
    }

    template <typename data_t>
    SQS<data_t>* SQS<data_t>::cloneImpl() const
    {
        std::vector<std::unique_ptr<L2NormPow2<data_t>>> newsubsets;
        newsubsets.reserve(subsets_.size());
        for (const auto& ptr : subsets_) {
            newsubsets.emplace_back(downcast<L2NormPow2<data_t>>(ptr->clone()));
        }

        if (preconditioner_)
            return new SQS(*fullProblem_, std::move(newsubsets), *preconditioner_,
                           momentumAcceleration_, epsilon_);

        return new SQS(*fullProblem_, std::move(newsubsets), momentumAcceleration_, epsilon_);
    }

    template <typename data_t>
    bool SQS<data_t>::isEqual(const Solver<data_t>& other) const
    {
        auto otherSQS = downcast_safe<SQS>(&other);
        if (!otherSQS)
            return false;

        if (epsilon_ != otherSQS->epsilon_)
            return false;

        if ((preconditioner_ && !otherSQS->preconditioner_)
            || (!preconditioner_ && otherSQS->preconditioner_))
            return false;

        if (preconditioner_ && otherSQS->preconditioner_)
            if (*preconditioner_ != *otherSQS->preconditioner_)
                return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class SQS<float>;
    template class SQS<double>;

} // namespace elsa
