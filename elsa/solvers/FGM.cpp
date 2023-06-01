#include "FGM.h"
#include "DataContainer.h"
#include "Error.h"
#include "Functional.h"
#include "Logger.h"
#include "TypeCasts.hpp"
#include "PowerIterations.h"

namespace elsa
{
    template <typename data_t>
    FGM<data_t>::FGM(const Functional<data_t>& problem, data_t epsilon)
        : Solver<data_t>(), _problem(problem.clone()), _epsilon{epsilon}
    {
        if (!problem.isDifferentiable()) {
            throw InvalidArgumentError("FGM: Given problem is not differentiable!");
        }
    }

    template <typename data_t>
    FGM<data_t>::FGM(const Functional<data_t>& problem,
                     const LinearOperator<data_t>& preconditionerInverse, data_t epsilon)
        : Solver<data_t>(),
          _problem(problem.clone()),
          _epsilon{epsilon},
          _preconditionerInverse{preconditionerInverse.clone()}
    {
        if (!problem.isDifferentiable()) {
            throw InvalidArgumentError("FGM: Given problem is not differentiable!");
        }

        // check that preconditioner is compatible with problem
        if (_preconditionerInverse->getDomainDescriptor().getNumberOfCoefficients()
                != _problem->getDomainDescriptor().getNumberOfCoefficients()
            || _preconditionerInverse->getRangeDescriptor().getNumberOfCoefficients()
                   != _problem->getDomainDescriptor().getNumberOfCoefficients()) {
            throw InvalidArgumentError("FGM: incorrect size of preconditioner");
        }
    }

    template <typename data_t>
    DataContainer<data_t> FGM<data_t>::solve(index_t iterations,
                                             std::optional<DataContainer<data_t>> x0)
    {
        auto x = extract_or(x0, _problem->getDomainDescriptor());

        auto thetaOld = static_cast<data_t>(1.0);
        auto yOld = x;

        auto deltaZero = _problem->getGradient(x).squaredL2Norm();
        auto L = powerIterations(_problem->getHessian(x), 5);

        auto y = emptylike(x);

        Logger::get("FGM")->info("Starting optimization with lipschitz constant {}", L);
        Logger::get("FGM")->info("| {:^4} | {:^13} | {:^13} |", "", "objective", "gradient");

        for (index_t i = 0; i < iterations; ++i) {
            auto gradient = _problem->getGradient(x);

            if (_preconditionerInverse)
                gradient = _preconditionerInverse->apply(gradient);

            lincomb(1, x, -1 / L, gradient, y);

            const auto theta =
                (data_t{1.0} + std::sqrt(data_t{1.0} + data_t{4.0} * thetaOld * thetaOld))
                / data_t{2.0};
            lincomb(1, y, (thetaOld - data_t{1}) / theta, (y - yOld), x);

            Logger::get("FGM")->info("| {:>4} | {:>13} | {:>13} |", i, _problem->evaluate(x),
                                     gradient.squaredL2Norm());

            thetaOld = theta;
            yOld = y;

            // if the gradient is too small we stop
            if (gradient.squaredL2Norm() <= _epsilon * _epsilon * deltaZero) {
                Logger::get("FGM")->info("SUCCESS: Reached convergence at {}/{} iteration", i + 1,
                                         iterations);
                return x;
            }
        }

        Logger::get("FGM")->warn("Failed to reach convergence at {} iterations", iterations);

        return x;
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
