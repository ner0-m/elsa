#include <iostream>
#include "OGM.h"
#include "Logger.h"

namespace elsa
{
    template <typename data_t>
    OGM<data_t>::OGM(const Problem<data_t>& problem, data_t epsilon)
        : Solver<data_t>(problem), _epsilon{epsilon}
    {
    }

    template <typename data_t>
    DataContainer<data_t>& OGM<data_t>::solveImpl(index_t iterations)
    {
        if (iterations == 0)
            iterations = _defaultIterations;

        auto prevTheta = static_cast<data_t>(1.0);
        auto x0 = DataContainer<data_t>(getCurrentSolution());
        auto& prevY = x0;

        auto lipschitz = _problem->getLipschitzConstant();
        auto deltaZero = _problem->getGradient().squaredL2Norm();
        Logger::get("OGM")->info("Starting optimization with lipschitz constant {}", lipschitz);

        for (index_t i = 0; i < iterations; ++i) {
            Logger::get("OGM")->info("iteration {} of {}", i + 1, iterations);
            auto& x = getCurrentSolution();

            auto gradient = _problem->getGradient();

            DataContainer<data_t> y = x - gradient / lipschitz;
            data_t theta;
            if (i == iterations - 1) { // last iteration
                theta = (static_cast<data_t>(1.0)
                         + std::sqrt(static_cast<data_t>(1.0)
                                     + static_cast<data_t>(8.0) * prevTheta * prevTheta))
                        / static_cast<data_t>(2.0);
            } else {
                theta = (static_cast<data_t>(1.0)
                         + std::sqrt(static_cast<data_t>(1.0)
                                     + static_cast<data_t>(4.0) * prevTheta * prevTheta))
                        / static_cast<data_t>(2.0);
            }
            // x_{i+1} = y_{i+1} + \frac{\theta_i-1}{\theta_{i+1}}(y_{i+1} - y_i) +
            // \frac{\theta_i}{\theta_{i+1}}/(y_{i+1} - x_i)
            x = y + (prevTheta - static_cast<data_t>(1.0)) / theta * (y - prevY)
                - prevTheta / theta * gradient / lipschitz;
            prevTheta = theta;
            prevY = y;

            // if the gradient is too small we stop
            if (gradient.squaredL2Norm() <= _epsilon * _epsilon * deltaZero) {
                Logger::get("OGM")->info("SUCCESS: Reached convergence at {}/{} iteration", i + 1,
                                         iterations);
                return x;
            }
        }

        Logger::get("OGM")->warn("Failed to reach convergence at {} iterations", iterations);

        return getCurrentSolution();
    }

    template <typename data_t>
    OGM<data_t>* OGM<data_t>::cloneImpl() const
    {
        return new OGM(*_problem, _epsilon);
    }

    template <typename data_t>
    bool OGM<data_t>::isEqual(const Solver<data_t>& other) const
    {
        if (!Solver<data_t>::isEqual(other))
            return false;

        auto otherOGM = dynamic_cast<const OGM*>(&other);
        if (!otherOGM)
            return false;

        if (_epsilon != otherOGM->_epsilon)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class OGM<float>;
    template class OGM<double>;

} // namespace elsa
