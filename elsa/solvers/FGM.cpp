#include <iostream>
#include "FGM.h"
#include "Logger.h"

namespace elsa
{
    template <typename data_t>
    FGM<data_t>::FGM(const Problem<data_t>& problem, data_t epsilon)
        : Solver<data_t>(problem), _epsilon{epsilon}
    {
    }

    template <typename data_t>
    DataContainer<data_t>& FGM<data_t>::solveImpl(index_t iterations)
    {
        if (iterations == 0)
            iterations = _defaultIterations;

        auto prevTheta = static_cast<data_t>(1.0);
        auto x0 = DataContainer<data_t>(getCurrentSolution());
        auto& prevY = x0;

        auto deltaZero = _problem->getGradient().squaredL2Norm();
        auto lipschitz = _problem->getLipschitzConstant();
        Logger::get("FGM")->info("Starting optimization with lipschitz constant {}", lipschitz);

        for (index_t i = 0; i < iterations; ++i) {
            Logger::get("FGM")->info("iteration {} of {}", i + 1, iterations);
            auto& x = getCurrentSolution();

            auto gradient = _problem->getGradient();

            DataContainer<data_t> y = x - gradient / lipschitz;
            const auto theta = (static_cast<data_t>(1.0)
                                + std::sqrt(static_cast<data_t>(1.0)
                                            + static_cast<data_t>(4.0) * prevTheta * prevTheta))
                               / static_cast<data_t>(2.0);
            x = y + (prevTheta - static_cast<data_t>(1.0)) / theta * (y - prevY);
            prevTheta = theta;
            prevY = y;

            // if the gradient is too small we stop
            if (gradient.squaredL2Norm() <= _epsilon * _epsilon * deltaZero) {
                Logger::get("FGM")->info("SUCCESS: Reached convergence at {}/{} iteration", i + 1,
                                         iterations);
                return x;
            }
        }

        Logger::get("FGM")->warn("Failed to reach convergence at {} iterations", iterations);

        return getCurrentSolution();
    }

    template <typename data_t>
    FGM<data_t>* FGM<data_t>::cloneImpl() const
    {
        return new FGM(*_problem, _epsilon);
    }

    template <typename data_t>
    bool FGM<data_t>::isEqual(const Solver<data_t>& other) const
    {
        if (!Solver<data_t>::isEqual(other))
            return false;

        auto otherFGM = dynamic_cast<const FGM*>(&other);
        if (!otherFGM)
            return false;

        if (_epsilon != otherFGM->_epsilon)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class FGM<float>;
    template class FGM<double>;

} // namespace elsa
