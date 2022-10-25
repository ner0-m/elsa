#include <Identity.h>
#include <Scaling.h>
#include "SQS.h"
#include "Logger.h"
#include "SubsetProblem.h"
#include "TypeCasts.hpp"

namespace elsa
{
    template <typename data_t>
    SQS<data_t>::SQS(const Problem<data_t>& problem, bool momentumAcceleration, data_t epsilon)
        : Solver<data_t>(),
          _problem(problem.clone()),
          _epsilon{epsilon},
          _momentumAcceleration{momentumAcceleration}
    {
        if (is<SubsetProblem<data_t>>(problem)) {
            Logger::get("SQS")->info(
                "SQS did received a SubsetProblem, running in ordered subset mode");
            _subsetMode = true;
        } else {
            Logger::get("SQS")->info("SQS did not receive a SubsetProblem, running in normal mode");
        }
    }

    template <typename data_t>
    SQS<data_t>::SQS(const Problem<data_t>& problem, const LinearOperator<data_t>& preconditioner,
                     bool momentumAcceleration, data_t epsilon)
        : Solver<data_t>(),
          _problem(problem.clone()),
          _epsilon{epsilon},
          _preconditioner{preconditioner.clone()},
          _momentumAcceleration{momentumAcceleration}
    {
        if (is<SubsetProblem<data_t>>(problem)) {
            Logger::get("SQS")->info(
                "SQS did received a SubsetProblem, running in ordered subset mode");
            _subsetMode = true;
        } else {
            Logger::get("SQS")->info("SQS did not receive a SubsetProblem, running in normal mode");
        }

        // check that preconditioner is compatible with problem
        if (_preconditioner->getDomainDescriptor().getNumberOfCoefficients()
                != _problem->getDataTerm().getDomainDescriptor().getNumberOfCoefficients()
            || _preconditioner->getRangeDescriptor().getNumberOfCoefficients()
                   != _problem->getDataTerm().getDomainDescriptor().getNumberOfCoefficients()) {
            throw InvalidArgumentError("SQS: incorrect size of preconditioner");
        }
    }

    template <typename data_t>
    DataContainer<data_t> SQS<data_t>::solve(index_t iterations,
                                             std::optional<DataContainer<data_t>> x0)
    {
        auto& solutionDesc = _problem->getDataTerm().getDomainDescriptor();
        auto x = DataContainer<data_t>(solutionDesc);
        if (x0.has_value()) {
            x = *x0;
        } else {
            x = 0;
        }

        auto convergenceThreshold = _problem->getGradient(x).squaredL2Norm() * _epsilon * _epsilon;

        auto hessian = _problem->getHessian(x);

        auto ones = DataContainer<data_t>(solutionDesc);
        ones = 1;
        auto diagVector = hessian->apply(ones);
        diagVector = static_cast<data_t>(1.0) / diagVector;
        auto diag = Scaling<data_t>(hessian->getDomainDescriptor(), diagVector);

        data_t prevT = 1;
        data_t t = 1;
        data_t nextT = 0;

        auto& z = x;
        // z = 0;
        DataContainer<data_t> prevX = x;
        DataContainer<data_t> gradient(solutionDesc);

        index_t nSubsets = 1;
        if (_subsetMode) {
            const auto& subsetProblem = static_cast<const SubsetProblem<data_t>*>(_problem.get());
            nSubsets = subsetProblem->getNumberOfSubsets();
        }

        for (index_t i = 0; i < iterations; i++) {
            Logger::get("SQS")->info("iteration {} of {}", i + 1, iterations);

            for (index_t m = 0; m < nSubsets; m++) {
                if (_subsetMode) {
                    gradient = static_cast<SubsetProblem<data_t>*>(_problem.get())
                                   ->getSubsetGradient(x, m);
                } else {
                    gradient = _problem->getGradient(x);
                }

                if (_preconditioner)
                    gradient = _preconditioner->apply(gradient);

                // TODO: element wise relu
                if (_momentumAcceleration) {
                    nextT = as<data_t>(1)
                            + std::sqrt(as<data_t>(1) + as<data_t>(4) * t * t) / as<data_t>(2);

                    x = z - nSubsets * diag.apply(gradient);
                    z = x + prevT / nextT * (x - prevX);
                } else {
                    z = z - nSubsets * diag.apply(gradient);
                }

                // if the gradient is too small we stop
                if (gradient.squaredL2Norm() <= convergenceThreshold) {
                    if (!_subsetMode
                        || _problem->getGradient(x).squaredL2Norm() <= convergenceThreshold) {
                        Logger::get("SQS")->info("SUCCESS: Reached convergence at {}/{} iteration",
                                                 i + 1, iterations);

                        // TODO: make return more sane
                        if (_momentumAcceleration) {
                            z = x;
                        }
                        return x;
                    }
                }

                if (_momentumAcceleration) {
                    prevT = t;
                    t = nextT;
                    prevX = x;
                }
            }
        }

        Logger::get("SQS")->warn("Failed to reach convergence at {} iterations", iterations);

        // TODO: make return more sane
        if (_momentumAcceleration) {
            z = x;
        }
        return x;
    }

    template <typename data_t>
    SQS<data_t>* SQS<data_t>::cloneImpl() const
    {
        if (_preconditioner)
            return new SQS(*_problem, *_preconditioner, _momentumAcceleration, _epsilon);

        return new SQS(*_problem, _momentumAcceleration, _epsilon);
    }

    template <typename data_t>
    bool SQS<data_t>::isEqual(const Solver<data_t>& other) const
    {
        auto otherSQS = downcast_safe<SQS>(&other);
        if (!otherSQS)
            return false;

        if (_epsilon != otherSQS->_epsilon)
            return false;

        if ((_preconditioner && !otherSQS->_preconditioner)
            || (!_preconditioner && otherSQS->_preconditioner))
            return false;

        if (_preconditioner && otherSQS->_preconditioner)
            if (*_preconditioner != *otherSQS->_preconditioner)
                return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class SQS<float>;
    template class SQS<double>;

} // namespace elsa
