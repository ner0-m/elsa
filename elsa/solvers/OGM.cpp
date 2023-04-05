#include "OGM.h"
#include "Functional.h"
#include "TypeCasts.hpp"
#include "Logger.h"
#include "PowerIterations.h"

namespace elsa
{
    template <typename data_t>
    OGM<data_t>::OGM(const Functional<data_t>& problem, data_t epsilon)
        : Solver<data_t>(), _problem(problem.clone()), _epsilon{epsilon}
    {
        if (!problem.isDifferentiable()) {
            throw InvalidArgumentError("OGM: Given problem is not differentiable!");
        }
    }

    template <typename data_t>
    OGM<data_t>::OGM(const Functional<data_t>& problem,
                     const LinearOperator<data_t>& preconditionerInverse, data_t epsilon)
        : Solver<data_t>(),
          _problem(problem.clone()),
          _epsilon{epsilon},
          _preconditionerInverse{preconditionerInverse.clone()}
    {
        if (!problem.isDifferentiable()) {
            throw InvalidArgumentError("OGM: Given problem is not differentiable!");
        }

        // check that preconditioner is compatible with problem
        if (_preconditionerInverse->getDomainDescriptor().getNumberOfCoefficients()
                != _problem->getDomainDescriptor().getNumberOfCoefficients()
            || _preconditionerInverse->getRangeDescriptor().getNumberOfCoefficients()
                   != _problem->getDomainDescriptor().getNumberOfCoefficients()) {
            throw InvalidArgumentError("OGM: incorrect size of preconditioner");
        }
    }

    template <typename data_t>
    DataContainer<data_t> OGM<data_t>::solve(index_t iterations,
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

        // OGM is very picky when it comes to the accuracy of the used lipschitz constant therefore
        // we use 20 power iterations instead of 5 here to be more precise.
        // In some cases OGM might still not converge then an even more precise constant is needed
        auto lipschitz = powerIterations(_problem->getHessian(x), 20);
        auto deltaZero = _problem->getGradient(x).squaredL2Norm();
        Logger::get("OGM")->info("Starting optimization with lipschitz constant {}", lipschitz);

        // log history legend
        Logger::get("OGM")->info("| {:^4} | {:^13} | {:^13} |", "", "objective", "gradient");

        for (index_t i = 0; i < iterations; ++i) {
            auto gradient = _problem->getGradient(x);

            if (_preconditionerInverse)
                gradient = _preconditionerInverse->apply(gradient);

            DataContainer<data_t> y = x - gradient / lipschitz;
            const auto f = (i == iterations - 1) ? data_t{8} : data_t{4};
            const auto theta =
                data_t{0.5} * (data_t{1} + std::sqrt(data_t{1} + f * std::pow(prevTheta, 2)));

            // x_{i+1} = y_{i+1} + \frac{\theta_i-1}{\theta_{i+1}}(y_{i+1} - y_i) +
            // \frac{\theta_i}{\theta_{i+1}}/(y_{i+1} - x_i)
            x = y + ((prevTheta - static_cast<data_t>(1.0)) / theta) * (y - prevY)
                - (prevTheta / theta) * (gradient / lipschitz);

            Logger::get("OGM")->info("| {:>4} | {:>13} | {:>13} |", i, _problem->evaluate(x),
                                     gradient.squaredL2Norm());

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

        return x;
    }

    template <typename data_t>
    OGM<data_t>* OGM<data_t>::cloneImpl() const
    {
        if (_preconditionerInverse)
            return new OGM(*_problem, *_preconditionerInverse, _epsilon);

        return new OGM(*_problem, _epsilon);
    }

    template <typename data_t>
    bool OGM<data_t>::isEqual(const Solver<data_t>& other) const
    {
        auto otherOGM = downcast_safe<OGM>(&other);
        if (!otherOGM)
            return false;

        if (_epsilon != otherOGM->_epsilon)
            return false;

        if ((_preconditionerInverse && !otherOGM->_preconditionerInverse)
            || (!_preconditionerInverse && otherOGM->_preconditionerInverse))
            return false;

        if (_preconditionerInverse && otherOGM->_preconditionerInverse)
            if (*_preconditionerInverse != *otherOGM->_preconditionerInverse)
                return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class OGM<float>;
    template class OGM<double>;

} // namespace elsa
