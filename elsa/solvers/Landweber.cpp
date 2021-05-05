#include "Landweber.h"
#include "Logger.h"

#include <stdexcept>

namespace elsa
{
    template <typename data_t>
    Landweber<data_t>::Landweber(const WLSProblem<data_t>& problem, data_t stepSize, bool projected)
        : Solver<data_t>(problem), _stepSize{stepSize}, projected{projected}
    {
        // sanity check
        if (_stepSize <= 0)
            throw std::invalid_argument("Landweber: step size has to be positive");
    }

    template <typename data_t>
    Landweber<data_t>::Landweber(const WLSProblem<data_t>& problem, bool projected)
        : Solver<data_t>(problem), projected{projected}
    {
        this->_stepSize =
            static_cast<data_t>(2.0) / static_cast<data_t>(problem.getLipschitzConstant())
            - std::numeric_limits<data_t>::epsilon();
    }

    template <typename data_t>
    Landweber<data_t>::Landweber(const WLSProblem<data_t>& problem)
        : Solver<data_t>(problem), projected(false)
    {
        this->_stepSize =
            static_cast<data_t>(2.0) / static_cast<data_t>(problem.getLipschitzConstant())
            - std::numeric_limits<data_t>::epsilon();
    }

    template <typename data_t>
    DataContainer<data_t>& Landweber<data_t>::solveImpl(index_t iterations)
    {
        if (iterations == 0)
            iterations = _defaultIterations;

        for (index_t i = 0; i < iterations; ++i) {
            Logger::get("Landweber algorithm")->info("iteration {} of {}", i + 1, iterations);
            auto& x = getCurrentSolution();

            auto gradient = _problem->getGradient();
            gradient *= _stepSize;
            x -= gradient;
            if (projected) {
                for (auto& elem : x)
                    elem = (elem < 0) ? static_cast<data_t>(0.0) : elem;
            }
        }

        return getCurrentSolution();
    }

    template <typename data_t>
    Landweber<data_t>* Landweber<data_t>::cloneImpl() const
    {
        return new Landweber(*(dynamic_cast<WLSProblem<data_t>*>(_problem.get())), _stepSize,
                             projected);
    }

    template <typename data_t>
    bool Landweber<data_t>::isEqual(const Solver<data_t>& other) const
    {
        if (!Solver<data_t>::isEqual(other))
            return false;

        auto otherGD = dynamic_cast<const Landweber*>(&other);
        if (!otherGD)
            return false;

        if (_stepSize != otherGD->_stepSize)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class Landweber<float>;
    template class Landweber<double>;
} // namespace elsa
