#include "FISTA.h"
#include "SoftThresholding.h"
#include "Logger.h"

#include "spdlog/stopwatch.h"

namespace elsa
{
    template <typename data_t>
    FISTA<data_t>::FISTA(const Problem<data_t>& problem, geometry::Threshold<data_t> mu,
                         data_t epsilon)
        : Solver<data_t>(LASSOProblem(problem)), _mu{mu}, _epsilon{epsilon}
    {
    }

    template <typename data_t>
    FISTA<data_t>::FISTA(const Problem<data_t>& problem, data_t epsilon)
        : FISTA<data_t>(LASSOProblem(problem), epsilon)
    {
    }

    template <typename data_t>
    FISTA<data_t>::FISTA(const LASSOProblem<data_t>& problem, data_t epsilon)
        : Solver<data_t>(LASSOProblem(problem)),
          _mu{1 / problem.getLipschitzConstant()},
          _epsilon{epsilon}
    {
    }

    template <typename data_t>
    auto FISTA<data_t>::solveImpl(index_t iterations) -> DataContainer<data_t>&
    {
        if (iterations == 0)
            iterations = _defaultIterations;

        spdlog::stopwatch aggregate_time;

        SoftThresholding<data_t> shrinkageOp{getCurrentSolution().getDataDescriptor()};

        data_t lambda = _problem->getRegularizationTerms()[0].getWeight();

        // Safe as long as only LinearResidual exits
        const auto& linResid =
            downcast<LinearResidual<data_t>>((_problem->getDataTerm()).getResidual());
        const LinearOperator<data_t>& A = linResid.getOperator();
        const DataContainer<data_t>& b = linResid.getDataVector();

        DataContainer<data_t>& x = getCurrentSolution();
        DataContainer<data_t> xPrev = getCurrentSolution();
        DataContainer<data_t> y = getCurrentSolution();
        DataContainer<data_t> yPrev = getCurrentSolution();
        data_t t;
        data_t tPrev = 1;

        DataContainer<data_t> Atb = A.applyAdjoint(b);
        DataContainer<data_t> gradient = A.applyAdjoint(A.apply(yPrev)) - Atb;

        Logger::get("FISTA")->info("{:^6}|{:*^16}|{:*^8}|{:*^8}|", "iter", "gradient", "time",
                                   "elapsed");

        auto deltaZero = gradient.squaredL2Norm();
        for (index_t iter = 0; iter < iterations; ++iter) {
            spdlog::stopwatch iter_time;

            gradient = A.applyAdjoint(A.apply(yPrev)) - Atb;
            x = shrinkageOp.apply(yPrev - _mu * gradient, geometry::Threshold{_mu * lambda});

            t = (1 + std::sqrt(1 + 4 * tPrev * tPrev)) / 2;
            y = x + ((tPrev - 1) / t) * (x - xPrev);

            xPrev = x;
            yPrev = y;
            tPrev = t;

            Logger::get("FISTA")->info("{:>5} |{:>15} | {:>6.3} |{:>6.3}s |", iter,
                                       gradient.squaredL2Norm(), iter_time, aggregate_time);

            if (gradient.squaredL2Norm() <= _epsilon * _epsilon * deltaZero) {
                Logger::get("FISTA")->info("SUCCESS: Reached convergence at {}/{} iteration",
                                           iter + 1, iterations);
                return x;
            }
        }

        Logger::get("FISTA")->warn("Failed to reach convergence at {} iterations", iterations);

        return getCurrentSolution();
    }

    template <typename data_t>
    auto FISTA<data_t>::cloneImpl() const -> FISTA<data_t>*
    {
        return new FISTA(*_problem, geometry::Threshold<data_t>{_mu}, _epsilon);
    }

    template <typename data_t>
    auto FISTA<data_t>::isEqual(const Solver<data_t>& other) const -> bool
    {
        if (!Solver<data_t>::isEqual(other))
            return false;

        auto otherFISTA = downcast_safe<FISTA>(&other);
        if (!otherFISTA)
            return false;

        if (_mu != otherFISTA->_mu)
            return false;

        if (_epsilon != otherFISTA->_epsilon)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class FISTA<float>;
    template class FISTA<double>;
} // namespace elsa
