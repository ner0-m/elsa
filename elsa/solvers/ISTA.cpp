#include "ISTA.h"
#include "SoftThresholding.h"
#include "TypeCasts.hpp"
#include "Logger.h"

#include "spdlog/stopwatch.h"

namespace elsa
{
    template <typename data_t>
    ISTA<data_t>::ISTA(const Problem<data_t>& problem, geometry::Threshold<data_t> mu,
                       data_t epsilon)
        : Solver<data_t>(LASSOProblem(problem)), _mu{mu}, _epsilon{epsilon}
    {
    }

    template <typename data_t>
    ISTA<data_t>::ISTA(const Problem<data_t>& problem, data_t epsilon)
        : ISTA<data_t>(LASSOProblem(problem), epsilon)
    {
    }

    template <typename data_t>
    ISTA<data_t>::ISTA(const LASSOProblem<data_t>& lassoProb, data_t epsilon)
        : Solver<data_t>(lassoProb), _mu{1 / lassoProb.getLipschitzConstant()}, _epsilon{epsilon}
    {
    }

    template <typename data_t>
    auto ISTA<data_t>::solveImpl(index_t iterations) -> DataContainer<data_t>&
    {
        if (iterations == 0)
            iterations = _defaultIterations;

        spdlog::stopwatch aggregate_time;
        Logger::get("ISTA")->info("Start preparations...");

        SoftThresholding<data_t> shrinkageOp{getCurrentSolution().getDataDescriptor()};

        data_t lambda = _problem->getRegularizationTerms()[0].getWeight();

        // Safe as long as only LinearResidual exits
        const auto& linResid =
            downcast<LinearResidual<data_t>>((_problem->getDataTerm()).getResidual());
        const LinearOperator<data_t>& A = linResid.getOperator();
        const DataContainer<data_t>& b = linResid.getDataVector();

        DataContainer<data_t>& x = getCurrentSolution();
        DataContainer<data_t> Atb = A.applyAdjoint(b);
        DataContainer<data_t> gradient = A.applyAdjoint(A.apply(x)) - Atb;

        Logger::get("ISTA")->info("Preparations done, tooke {}s", aggregate_time);
        Logger::get("ISTA")->info("{:^6}|{:*^16}|{:*^8}|{:*^8}|", "iter", "gradient", "time",
                                  "elapsed");

        auto deltaZero = gradient.squaredL2Norm();
        for (index_t iter = 0; iter < iterations; ++iter) {
            spdlog::stopwatch iter_time;

            gradient = A.applyAdjoint(A.apply(x)) - Atb;

            x = shrinkageOp.apply(x - _mu * gradient, geometry::Threshold{_mu * lambda});

            Logger::get("ISTA")->info("{:>5} |{:>15} | {:>6.3} |{:>6.3}s |", iter,
                                      gradient.squaredL2Norm(), iter_time, aggregate_time);
            if (gradient.squaredL2Norm() <= _epsilon * _epsilon * deltaZero) {
                Logger::get("ISTA")->info("SUCCESS: Reached convergence at {}/{} iteration",
                                          iter + 1, iterations);
                return x;
            }
        }

        Logger::get("ISTA")->warn("Failed to reach convergence at {} iterations", iterations);

        return getCurrentSolution();
    }

    template <typename data_t>
    auto ISTA<data_t>::cloneImpl() const -> ISTA<data_t>*
    {
        return new ISTA(*_problem, geometry::Threshold<data_t>{_mu}, _epsilon);
    }

    template <typename data_t>
    auto ISTA<data_t>::isEqual(const Solver<data_t>& other) const -> bool
    {
        if (!Solver<data_t>::isEqual(other))
            return false;

        auto otherISTA = downcast_safe<ISTA>(&other);
        if (!otherISTA)
            return false;

        if (_mu != otherISTA->_mu)
            return false;

        if (_epsilon != otherISTA->_epsilon)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class ISTA<float>;
    template class ISTA<double>;
} // namespace elsa
