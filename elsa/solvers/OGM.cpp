#include "OGM.h"
#include "TypeCasts.hpp"
#include "Logger.h"

namespace elsa
{
    template <typename data_t>
    OGM<data_t>::OGM(const Problem<data_t>& problem, data_t epsilon)
        : Solver<data_t>(), _problem(problem.clone()), _epsilon{epsilon}
    {
    }

    template <typename data_t>
    OGM<data_t>::OGM(const Problem<data_t>& problem,
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
            throw InvalidArgumentError("OGM: incorrect size of preconditioner");
        }
    }

    template <typename data_t>
    DataContainer<data_t>& OGM<data_t>::solveImpl(index_t iterations)
    {
        if (iterations == 0)
            iterations = _defaultIterations;

        auto prevTheta = static_cast<data_t>(1.0);
        auto x0 = _problem->getCurrentSolution();
        auto& prevY = x0;

        // OGM is very picky when it comes to the accuracy of the used lipschitz constant therefore
        // we use 20 power iterations instead of 5 here to be more precise.
        // In some cases OGM might still not converge then an even more precise constant is needed
        auto lipschitz = _problem->getLipschitzConstant(20);
        auto deltaZero = _problem->getGradient().squaredL2Norm();
        Logger::get("OGM")->info("Starting optimization with lipschitz constant {}", lipschitz);

        // log history legend
        Logger::get("OGM")->info("{:*^20}|{:*^20}|{:*^20}|{:*^20}|{:*^20}", "iteration",
                                 "thetaRatio0", "thetaRatio1", "y", "gradient");

        for (index_t i = 0; i < iterations; ++i) {
            auto& x = _problem->getCurrentSolution();

            auto gradient = _problem->getGradient();

            if (_preconditionerInverse)
                gradient = _preconditionerInverse->apply(gradient);

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

            Logger::get("OGM")->info(" {:<19}| {:<19}| {:<19}| {:<19}| {:<19}", i,
                                     (prevTheta - 1) / theta, prevTheta / theta, y.squaredL2Norm(),
                                     gradient.squaredL2Norm());

            // x_{i+1} = y_{i+1} + \frac{\theta_i-1}{\theta_{i+1}}(y_{i+1} - y_i) +
            // \frac{\theta_i}{\theta_{i+1}}/(y_{i+1} - x_i)
            x = y + ((prevTheta - static_cast<data_t>(1.0)) / theta) * (y - prevY)
                - (prevTheta / theta) * (gradient / lipschitz);
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

        return _problem->getCurrentSolution();
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
