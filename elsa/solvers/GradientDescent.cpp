#include "GradientDescent.h"
#include "Functional.h"
#include "Logger.h"
#include "TypeCasts.hpp"
#include "PowerIterations.h"
#include <iostream>

namespace elsa
{

    template <typename data_t>
    GradientDescent<data_t>::GradientDescent(const Functional<data_t>& problem, data_t stepSize)
        : Solver<data_t>(), _problem(problem.clone()), _stepSize{stepSize}
    {
        // sanity check
        if (stepSize <= 0)
            throw InvalidArgumentError("GradientDescent: step size has to be positive");
    }

    template <typename data_t>
    GradientDescent<data_t>::GradientDescent(const Functional<data_t>& problem)
        : Solver<data_t>(), _problem(problem.clone())
    {
    }

    template <typename data_t>
    DataContainer<data_t> GradientDescent<data_t>::solve(index_t iterations,
                                                         std::optional<DataContainer<data_t>> x0)
    {
        auto x = DataContainer<data_t>(_problem->getDomainDescriptor());
        if (x0.has_value()) {
            x = *x0;
        } else {
            x = 0;
        }

        // If stepSize is not initialized yet, we do it know with x0
        if (!_stepSize.isInitialized()) {
            _stepSize = powerIterations(_problem->getHessian(x));
            Logger::get("GradientDescent")
                ->info("Step length is chosen to be: {:8.5})", *_stepSize);
        }

        for (index_t i = 0; i < iterations; ++i) {
            Logger::get("GradientDescent")->info("iteration {} of {}", i + 1, iterations);
            auto gradient = _problem->getGradient(x);
            x -= *_stepSize * gradient;
        }

        return x;
    }

    template <typename data_t>
    GradientDescent<data_t>* GradientDescent<data_t>::cloneImpl() const
    {
        if (_stepSize.isInitialized()) {
            return new GradientDescent(*_problem, *_stepSize);
        } else {
            return new GradientDescent(*_problem);
        }
    }

    template <typename data_t>
    bool GradientDescent<data_t>::isEqual(const Solver<data_t>& other) const
    {
        auto otherGD = downcast_safe<GradientDescent<data_t>>(&other);
        if (!otherGD)
            return false;

        return _stepSize == otherGD->_stepSize;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class GradientDescent<float>;
    template class GradientDescent<double>;
} // namespace elsa
