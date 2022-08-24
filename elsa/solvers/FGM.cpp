#include "FGM.h"
#include "Logger.h"
#include "TypeCasts.hpp"

namespace elsa
{
    template <typename data_t>
    FGM<data_t>::FGM(const Problem<data_t>& problem, data_t epsilon)
        : Solver<data_t>(), _problem(problem.clone()), _epsilon{epsilon}
    {
    }

    template <typename data_t>
    FGM<data_t>::FGM(const Problem<data_t>& problem,
                     const LinearOperator<data_t>& preconditionerInverse, data_t epsilon)
        : Solver<data_t>(),
          _problem(problem.clone()),
          _epsilon{epsilon},
          _preconditionerInverse{preconditionerInverse.clone()}
    {
        // check that preconditioner is compatible with problem
        if (_preconditionerInverse->getDomainDescriptor().getNumberOfCoefficients()
                != _problem->getCurrentSolution().getSize()
            || _preconditionerInverse->getRangeDescriptor().getNumberOfCoefficients()
                   != _problem->getCurrentSolution().getSize()) {
            throw InvalidArgumentError("FGM: incorrect size of preconditioner");
        }
    }

    template <typename data_t>
    DataContainer<data_t>& FGM<data_t>::solveImpl(index_t iterations)
    {
        if (iterations == 0)
            iterations = _defaultIterations;

        auto prevTheta = static_cast<data_t>(1.0);
        auto x0 = _problem->getCurrentSolution();
        auto& prevY = x0;

        auto deltaZero = _problem->getGradient().squaredL2Norm();
        auto lipschitz = _problem->getLipschitzConstant();
        Logger::get("FGM")->info("Starting optimization with lipschitz constant {}", lipschitz);

        for (index_t i = 0; i < iterations; ++i) {
            Logger::get("FGM")->info("iteration {} of {}", i + 1, iterations);
            auto& x = _problem->getCurrentSolution();

            auto gradient = _problem->getGradient();

            if (_preconditionerInverse)
                gradient = _preconditionerInverse->apply(gradient);

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

        return _problem->getCurrentSolution();
    }

    template <typename data_t>
    FGM<data_t>* FGM<data_t>::cloneImpl() const
    {
        if (_preconditionerInverse)
            return new FGM(*_problem, *_preconditionerInverse, _epsilon);

        return new FGM(*_problem, _epsilon);
    }

    template <typename data_t>
    bool FGM<data_t>::isEqual(const Solver<data_t>& other) const
    {
        auto otherFGM = downcast_safe<FGM>(&other);
        if (!otherFGM)
            return false;

        if (_epsilon != otherFGM->_epsilon)
            return false;

        if ((_preconditionerInverse && !otherFGM->_preconditionerInverse)
            || (!_preconditionerInverse && otherFGM->_preconditionerInverse))
            return false;

        if (_preconditionerInverse && otherFGM->_preconditionerInverse)
            if (*_preconditionerInverse != *otherFGM->_preconditionerInverse)
                return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class FGM<float>;
    template class FGM<double>;

} // namespace elsa
