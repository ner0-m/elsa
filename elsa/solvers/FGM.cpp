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
                != _problem->getDataTerm().getDomainDescriptor().getNumberOfCoefficients()
            || _preconditionerInverse->getRangeDescriptor().getNumberOfCoefficients()
                   != _problem->getDataTerm().getDomainDescriptor().getNumberOfCoefficients()) {
            throw InvalidArgumentError("FGM: incorrect size of preconditioner");
        }
    }

    template <typename data_t>
    DataContainer<data_t> FGM<data_t>::solveImpl(index_t iterations)
    {
        auto prevTheta = static_cast<data_t>(1.0);
        auto x = DataContainer<data_t>(_problem->getDataTerm().getDomainDescriptor());
        x = 0;
        auto prevY = x;

        auto deltaZero = _problem->getGradient(x).squaredL2Norm();
        auto lipschitz = _problem->getLipschitzConstant(x);
        Logger::get("FGM")->info("Starting optimization with lipschitz constant {}", lipschitz);

        for (index_t i = 0; i < iterations; ++i) {
            Logger::get("FGM")->info("iteration {} of {}", i + 1, iterations);

            auto gradient = _problem->getGradient(x);

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
