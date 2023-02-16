#include "FGM.h"
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
    }

    template <typename data_t>
    FGM<data_t>::FGM(const Functional<data_t>& problem,
                     const LinearOperator<data_t>& preconditionerInverse, data_t epsilon)
        : Solver<data_t>(),
          _problem(problem.clone()),
          _epsilon{epsilon},
          _preconditionerInverse{preconditionerInverse.clone()}
    {
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
        auto prevTheta = static_cast<data_t>(1.0);
        auto x = DataContainer<data_t>(_problem->getDomainDescriptor());
        if (x0.has_value()) {
            x = *x0;
        } else {
            x = 0;
        }
        auto prevY = x;

        auto deltaZero = _problem->getGradient(x).squaredL2Norm();
        auto lipschitz = powerIterations(_problem->getHessian(x), 5);

        Logger::get("FGM")->info("Starting optimization with lipschitz constant {}", lipschitz);
        Logger::get("FGM")->info("| {:^4} | {:^13} | {:^13} |", "", "objective", "gradient");

        for (index_t i = 0; i < iterations; ++i) {
            auto gradient = _problem->getGradient(x);

            if (_preconditionerInverse)
                gradient = _preconditionerInverse->apply(gradient);

            DataContainer<data_t> y = x - gradient / lipschitz;
            const auto theta =
                (data_t{1.0} + std::sqrt(data_t{1.0} + data_t{4.0} * prevTheta * prevTheta))
                / data_t{2.0};
            x = y + (prevTheta - data_t{1.0}) / theta * (y - prevY);

            Logger::get("FGM")->info("| {:>4} | {:>13} | {:>13} |", i, _problem->evaluate(x),
                                     gradient.squaredL2Norm());

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
