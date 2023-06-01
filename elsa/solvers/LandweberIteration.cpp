#include "LandweberIteration.h"
#include "DataContainer.h"
#include "Functional.h"
#include "LinearOperator.h"
#include "Logger.h"
#include "TypeCasts.hpp"
#include "PowerIterations.h"
#include <iostream>

namespace elsa
{
    template <typename data_t>
    LandweberIteration<data_t>::LandweberIteration(const LinearOperator<data_t>& A,
                                                   const DataContainer<data_t>& b, data_t stepSize)
        : Solver<data_t>(), A_(A.clone()), b_(b), stepSize_{stepSize}
    {
    }

    template <typename data_t>
    LandweberIteration<data_t>::LandweberIteration(const LinearOperator<data_t>& A,
                                                   const DataContainer<data_t>& b)
        : Solver<data_t>(), A_(A.clone()), b_(b)
    {
    }

    template <typename data_t>
    DataContainer<data_t> LandweberIteration<data_t>::solve(index_t iterations,
                                                            std::optional<DataContainer<data_t>> x0)
    {
        auto x = extract_or(x0, A_->getDomainDescriptor());

        // We cannot call a virtual function in the constructor, so call it here
        if (!tam_) {
            tam_ = setupOperators(*A_);
        }

        if (!stepSize_.isInitialized()) {
            // Choose step length to be just below \f$\frac{2}{\sigma^2}\f$, where \f$\sigma\f$
            // is the largest eigenvalue of \f$T * A^T * M * A\f$. This is computed using the power
            // iterations.
            auto Anorm = powerIterations(*tam_ * *A_);
            stepSize_ = 0.9 * (2. / Anorm);
        }

        Logger::get("LandweberIterations")->info("Using Steplength: {}", *stepSize_);
        Logger::get("LandweberIterations")
            ->info(" {:^7} | {:^12} | {:^12} |", "Iters", "Recon", "Residual");

        auto residual = empty<data_t>(A_->getRangeDescriptor());
        for (index_t i = 0; i < iterations; ++i) {
            // Compute Ax - b memory efficient
            A_->apply(x, residual);
            residual -= b_;

            x -= (*stepSize_) * tam_->apply(residual);
            projection_(x);

            Logger::get("LandweberIterations")
                ->info(" {:>3}/{:>3} | {:>12.5f} | {:>12.5f} |", i + 1, iterations, x.l2Norm(),
                       residual.l2Norm());
        }

        return x;
    }

    template <typename data_t>
    bool LandweberIteration<data_t>::isEqual(const Solver<data_t>& other) const
    {
        auto landweber = downcast_safe<LandweberIteration<data_t>>(&other);
        return landweber && *A_ == *landweber->A_ && b_ == landweber->b_
               && stepSize_ == landweber->stepSize_;
    }

    template <class data_t>
    void LandweberIteration<data_t>::setProjection(
        const std::function<void(DataContainer<data_t>&)> projection)
    {
        projection_ = projection;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class LandweberIteration<float>;
    template class LandweberIteration<double>;
} // namespace elsa
