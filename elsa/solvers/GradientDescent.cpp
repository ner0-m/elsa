#include "GradientDescent.h"
#include "Logger.h"
#include "TypeCasts.hpp"
#include <iostream>

namespace elsa
{

    template <typename data_t>
    GradientDescent<data_t>::GradientDescent(const Problem<data_t>& problem, data_t stepSize)
        : Solver<data_t>(), _problem(problem.clone()), _stepSize{stepSize}
    {
        // sanity check
        if (stepSize <= 0)
            throw InvalidArgumentError("GradientDescent: step size has to be positive");
    }

    template <typename data_t>
    GradientDescent<data_t>::GradientDescent(const Problem<data_t>& problem)
        : Solver<data_t>(), _problem(problem.clone())
    {
    }

    template <typename data_t>
    DataContainer<data_t> GradientDescent<data_t>::solve(index_t iterations)
    {
        auto x = DataContainer<data_t>(_problem->getDataTerm().getDomainDescriptor());
        x = 0;

        // If stepSize is not initialized yet, we do it know with x0
        if (!_stepSize.isInitialized()) {
            _stepSize = static_cast<data_t>(1.0) / _problem->getLipschitzConstant(x);
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
