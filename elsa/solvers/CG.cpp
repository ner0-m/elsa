#include "CG.h"
#include "Logger.h"
#include "TypeCasts.hpp"

namespace elsa
{

    template <typename data_t>
    CG<data_t>::CG(const Problem<data_t>& problem, data_t epsilon)
        : Solver<data_t>{QuadricProblem<data_t>{problem}}, _epsilon{epsilon}
    {
    }

    template <typename data_t>
    CG<data_t>::CG(const Problem<data_t>& problem,
                   const LinearOperator<data_t>& preconditionerInverse, data_t epsilon)
        : Solver<data_t>{QuadricProblem<data_t>{problem}},
          _preconditionerInverse{preconditionerInverse.clone()},
          _epsilon{epsilon}
    {
        // check that preconditioner is compatible with problem
        if (_preconditionerInverse->getDomainDescriptor().getNumberOfCoefficients()
                != _problem->getCurrentSolution().getSize()
            || _preconditionerInverse->getRangeDescriptor().getNumberOfCoefficients()
                   != _problem->getCurrentSolution().getSize()) {
            throw InvalidArgumentError("CG: incorrect size of preconditioner");
        }
    }

    template <typename data_t>
    DataContainer<data_t>& CG<data_t>::solveImpl(index_t iterations)
    {
        if (iterations == 0)
            iterations = _defaultIterations;

        // get references to some variables in the Quadric
        auto& x = _problem->getCurrentSolution();
        const auto& gradientExpr =
            static_cast<const Quadric<data_t>&>(_problem->getDataTerm()).getGradientExpression();
        const LinearOperator<data_t>* A = nullptr;
        const DataContainer<data_t>* b = nullptr;

        if (gradientExpr.hasOperator())
            A = &gradientExpr.getOperator();

        if (gradientExpr.hasDataVector())
            b = &gradientExpr.getDataVector();

        // Start CG initialization
        auto r = _problem->getGradient();
        r *= static_cast<data_t>(-1.0);

        auto d = _preconditionerInverse ? _preconditionerInverse->apply(r) : r;

        // only allocate space for s if preconditioned
        std::unique_ptr<DataContainer<data_t>> s{};
        if (_preconditionerInverse)
            s = std::make_unique<DataContainer<data_t>>(
                _preconditionerInverse->getRangeDescriptor());

        auto deltaNew = r.dot(d);
        auto deltaZero = deltaNew;

        // log history legend
        Logger::get("CG")->info("{:*^20}|{:*^20}|{:*^20}|{:*^20}|{:*^20}", "iteration", "deltaNew",
                                "deltaZero", "epsilon", "objval");

        for (index_t it = 0; it != iterations; ++it) {
            auto Ad = A ? A->apply(d) : d;

            data_t alpha = deltaNew / d.dot(Ad);

            x += alpha * d;
            r -= alpha * Ad;

            if (_preconditionerInverse)
                _preconditionerInverse->apply(r, *s);

            const auto deltaOld = deltaNew;

            deltaNew = _preconditionerInverse ? r.dot(*s) : r.squaredL2Norm();

            // evaluate objective function as -0.5 * x^t[b + (b - Ax)]
            data_t objVal;
            if (b == nullptr) {
                objVal = static_cast<data_t>(-0.5) * x.dot(r);
            } else {
                objVal = static_cast<data_t>(-0.5) * x.dot(*b + r);
            }

            Logger::get("CG")->info(" {:<19}| {:<19}| {:<19}| {:<19}| {:<19}", it,
                                    std::sqrt(deltaNew), std::sqrt(deltaZero), _epsilon, objVal);

            if (deltaNew <= _epsilon * _epsilon * deltaZero) {
                // check that we are not stopping prematurely due to accumulated roundoff error
                r = _problem->getGradient();
                deltaNew = r.squaredL2Norm();
                if (deltaNew <= _epsilon * _epsilon * deltaZero) {
                    Logger::get("CG")->info("SUCCESS: Reached convergence at {}/{} iteration",
                                            it + 1, iterations);
                    return x;
                } else {
                    // we are very close to the desired solution, so do a hard reset
                    r *= static_cast<data_t>(-1.0);
                    d = 0;
                    if (_preconditionerInverse)
                        _preconditionerInverse->apply(r, *s);
                }
            }

            const auto beta = deltaNew / deltaOld;
            d = beta * d + (_preconditionerInverse ? *s : r);
        }

        Logger::get("CG")->warn("Failed to reach convergence at {} iterations", iterations);

        return x;
    }

    template <typename data_t>
    CG<data_t>* CG<data_t>::cloneImpl() const
    {
        if (_preconditionerInverse)
            return new CG(*_problem, *_preconditionerInverse, _epsilon);
        else
            return new CG(*_problem, _epsilon);
    }

    template <typename data_t>
    bool CG<data_t>::isEqual(const Solver<data_t>& other) const
    {
        if (!Solver<data_t>::isEqual(other))
            return false;

        auto otherCG = downcast_safe<CG>(&other);
        if (!otherCG)
            return false;

        if (_epsilon != otherCG->_epsilon)
            return false;

        if ((_preconditionerInverse && !otherCG->_preconditionerInverse)
            || (!_preconditionerInverse && otherCG->_preconditionerInverse))
            return false;

        if (_preconditionerInverse && otherCG->_preconditionerInverse)
            if (*_preconditionerInverse != *otherCG->_preconditionerInverse)
                return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class CG<float>;
    template class CG<double>;

} // namespace elsa
