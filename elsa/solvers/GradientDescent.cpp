#include "GradientDescent.h"
#include "Logger.h"

#include <stdexcept>

namespace elsa
{
    template <typename data_t>
    GradientDescent<data_t>::GradientDescent(const Problem<data_t>& problem, data_t stepSize)
        : Solver<data_t>(problem), _stepSize{stepSize}
    {
        // sanity check
        if (_stepSize <= 0)
            throw InvalidArgumentError("GradientDescent: step size has to be positive");
    }

    template <typename data_t>
    GradientDescent<data_t>::GradientDescent(const Problem<data_t>& problem)
        : Solver<data_t>(problem)
    {
        this->_stepSize =
            static_cast<data_t>(1.0) / static_cast<data_t>(problem.getLipschitzConstant());
    }

    template <typename data_t>
    DataContainer<data_t>& GradientDescent<data_t>::solveImpl(index_t iterations)
    {
        if (iterations == 0)
            iterations = _defaultIterations;

        for (index_t i = 0; i < iterations; ++i) {
            Logger::get("GradientDescent")->info("iteration {} of {}", i + 1, iterations);
            auto& x = getCurrentSolution();

            auto gradient = _problem->getGradient();
            gradient *= _stepSize;
            x -= gradient;
        }

        return getCurrentSolution();
    }

    template <typename data_t>
    GradientDescent<data_t>* GradientDescent<data_t>::cloneImpl() const
    {
        return new GradientDescent(*_problem, _stepSize);
    }

    template <typename data_t>
    bool GradientDescent<data_t>::isEqual(const Solver<data_t>& other) const
    {
        if (!Solver<data_t>::isEqual(other))
            return false;

        auto otherGD = dynamic_cast<const GradientDescent*>(&other);
        if (!otherGD)
            return false;

        if (_stepSize != otherGD->_stepSize)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class GradientDescent<float>;
    template class GradientDescent<double>;
} // namespace elsa
