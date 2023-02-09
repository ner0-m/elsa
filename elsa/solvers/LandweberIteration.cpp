#include "LandweberIteration.h"
#include "DataContainer.h"
#include "LinearOperator.h"
#include "Logger.h"
#include "TypeCasts.hpp"
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
    LandweberIteration<data_t>::LandweberIteration(const WLSProblem<data_t>& wls, data_t stepSize)
        : Solver<data_t>(), b_(wls.getDataTerm().getDomainDescriptor()), stepSize_{stepSize}
    {
        // sanity check
        if (stepSize <= 0)
            throw InvalidArgumentError("LandweberIteration: step size has to be positive");

        const auto& dataterm = wls.getDataTerm();
        const auto& residual = downcast_safe<LinearResidual<data_t>>(dataterm.getResidual());

        if (!residual.hasOperator()) {
            throw InvalidArgumentError(
                "LandweberIteration: WLSProblem passed to requieres an operator");
        }

        if (!residual.hasDataVector()) {
            throw InvalidArgumentError(
                "LandweberIteration: WLSProblem passed to requieres a data vector");
        }

        A_ = residual.getOperator().clone();
        b_ = residual.getDataVector();
    }

    template <typename data_t>
    LandweberIteration<data_t>::LandweberIteration(const WLSProblem<data_t>& wls)
        : Solver<data_t>(), b_(wls.getDataTerm().getDomainDescriptor())
    {
        const auto& dataterm = wls.getDataTerm();
        const auto& residual = downcast_safe<LinearResidual<data_t>>(dataterm.getResidual());

        if (!residual.hasOperator()) {
            throw InvalidArgumentError(
                "LandweberIteration: WLSProblem passed to requieres an operator");
        }

        if (!residual.hasDataVector()) {
            throw InvalidArgumentError(
                "LandweberIteration: WLSProblem passed to requieres a data vector");
        }

        A_ = residual.getOperator().clone();
        b_ = residual.getDataVector();
    }

    template <typename data_t>
    DataContainer<data_t> LandweberIteration<data_t>::solve(index_t iterations,
                                                            std::optional<DataContainer<data_t>> x0)
    {
        auto x = [&]() {
            if (x0.has_value()) {
                return *x0;
            } else {
                auto x = DataContainer<data_t>(A_->getDomainDescriptor());
                x = 0;
                return x;
            }
        }();

        // We cannot call a virtual function in the constructor, so call it here
        if (!tam_) {
            tam_ = setupOperators(*A_);
        }

        if (!stepSize_.isInitialized()) {
            stepSize_ = 1;
        }

        Logger::get("LandweberIterations")->info(" {:^7} | {:^12} |", "Iters", "Residual");

        auto residual = DataContainer<data_t>(A_->getRangeDescriptor());
        auto tmpx = DataContainer<data_t>(A_->getDomainDescriptor());
        for (index_t i = 0; i < iterations; ++i) {
            // Compute Ax - b memory efficient
            A_->apply(x, residual);
            residual -= b_;

            x -= (*stepSize_) * tam_->apply(residual);
            projection_(x);

            Logger::get("LandweberIterations")
                ->info(" {:>3}/{:>3} | {:>12.5f} |", i + 1, iterations, residual.l2Norm());
        }

        return x;
    }

    template <typename data_t>
    std::unique_ptr<LinearOperator<data_t>>
        LandweberIteration<data_t>::setupOperators(const WLSProblem<data_t>& wls) const
    {

        const auto& dataterm = wls.getDataTerm();
        const auto& residual = downcast_safe<LinearResidual<data_t>>(dataterm.getResidual());

        const auto& domain = residual.getDomainDescriptor();
        const auto& range = residual.getRangeDescriptor();

        const auto& A = residual.getOperator();

        return setupOperators(A);
    }

    template <typename data_t>
    bool LandweberIteration<data_t>::isEqual(const Solver<data_t>& other) const
    {
        auto otherSolver = downcast_safe<LandweberIteration<data_t>>(&other);
        if (!otherSolver)
            return false;

        return stepSize_ == otherSolver->stepSize_ && tam_ == otherSolver->tam_;
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
