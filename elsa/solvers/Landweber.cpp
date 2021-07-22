#include "Landweber.h"
#include "Logger.h"

#include <stdexcept>

namespace elsa
{
    template <typename data_t>
    Landweber<data_t>::Landweber(const WLSProblem<data_t>& problem, data_t stepSize,
                                 Projected projected)
        : Solver<data_t>(problem), _stepSize{stepSize}, _projected{static_cast<bool>(projected)}
    {
        // sanity check
        if (_stepSize <= 0)
            throw std::invalid_argument("Landweber: step size has to be positive");

        auto linResid =
            dynamic_cast<const LinearResidual<data_t>*>(&(_problem->getDataTerm()).getResidual());

        if (!linResid)
            throw std::logic_error("Landweber: Can only handle residuals of type 'LinearResidual'");
    }

    template <typename data_t>
    Landweber<data_t>::Landweber(const WLSProblem<data_t>& problem, Projected projected)
        : Landweber(problem, static_cast<data_t>(1.0) / problem.getLipschitzConstant(), projected)
    {
    }

    template <typename data_t>
    Landweber<data_t>::Landweber(const WLSProblem<data_t>& problem)
        : Landweber(problem, Projected::NO)
    {
    }

    template <typename data_t>
    DataContainer<data_t>& Landweber<data_t>::solveImpl(index_t iterations)
    {
        if (iterations == 0)
            iterations = _defaultIterations;
        auto& x = getCurrentSolution();
        for (index_t i = 0; i < iterations; ++i) {
            Logger::get("Landweber algorithm")->info("iteration {} of {}", i + 1, iterations);
            // add step size log
            auto gradient = _problem->getGradient();
            gradient *= _stepSize;
            x -= gradient;
            if (_projected) {
                for (auto& elem : x)
                    elem = (elem < 0) ? static_cast<data_t>(0.0) : elem;
            }
        }
        return getCurrentSolution();
    }

    template <typename data_t>
    Landweber<data_t>* Landweber<data_t>::cloneImpl() const
    {
        return new Landweber(*(static_cast<WLSProblem<data_t>*>(_problem.get())), _stepSize,
                             static_cast<Projected>(_projected));
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
