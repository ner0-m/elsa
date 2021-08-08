#include <BlockDescriptor.h>
#include "NLCG.h"
#include "Logger.h"
#include "Huber.h"

namespace elsa
{
    template <typename data_t>
    NLCG<data_t>::NLCG(const Problem<data_t>& problem, Beta beta)
        : Solver<data_t>(problem), _beta(beta)
    {
        auto linResid =
            dynamic_cast<const LinearResidual<data_t>*>(&(_problem->getDataTerm()).getResidual());

        if (!linResid)
            throw std::logic_error("NLCG: Can only handle residuals of type 'LinearResidual'");
    }

    template <typename data_t>
    NLCG<data_t>::NLCG(const Problem<data_t>& problem,
                       const LinearOperator<data_t>& preconditionerInverse)
        : Solver<data_t>{problem}, // remove the cast -> just accept problem
          _preconditionerInverse{preconditionerInverse.clone()}
    {
        // check that preconditioner is compatible with problem
        if (_preconditionerInverse->getDomainDescriptor().getNumberOfCoefficients()
                != _problem->getCurrentSolution().getSize()
            || _preconditionerInverse->getRangeDescriptor().getNumberOfCoefficients()
                   != _problem->getCurrentSolution().getSize()) {
            throw std::invalid_argument("NLCG: incorrect size of preconditioner");
        }
    }
    template <typename data_t>
    DataContainer<data_t>& NLCG<data_t>::solveImpl(index_t iterations)
    {
        if (iterations == 0)
            iterations = _defaultIterations;

        // get references to some variables
        auto& x = _problem->getCurrentSolution();
        // not quadric -> dynamic cast to linear residual
        // result von dataterm cast to linear residual -> as in SIRT
        const auto& gradientExpr =
            downcast<LinearResidual<data_t>>((_problem->getDataTerm()).getResidual());

        //  const auto& gradientExpr =
        //      static_cast<const
        //      Quadric<data_t>&>(_problem->getDataTerm()).getGradientExpression();

        const LinearOperator<data_t>* A = nullptr;
        const DataContainer<data_t>* b = nullptr;

        if (gradientExpr.hasOperator())
            A = &gradientExpr.getOperator();

        if (gradientExpr.hasDataVector())
            b = &gradientExpr.getDataVector();

        // Start CG initialization
        auto r = _problem->getGradient();
        r *= static_cast<data_t>(-1.0);
        auto d = r;

        auto deltaNew = r.dot(d); // squared l2
        auto deltaZero = deltaNew;
        data_t alpha;
        data_t epsilon = std::numeric_limits<data_t>::epsilon();
        for (index_t i = 0; i < iterations; ++i) {
            int j = 0;
            auto deltaD = d.squaredL2Norm();
            const auto rOld = r;

            do {
                auto numerator = _problem->getGradient().dot(d);
                numerator *= static_cast<data_t>(-1.0);
                auto denominator = d.dot(_problem->getHessian().apply(d));

                if (denominator == 0)
                    throw std::logic_error("NLCG: Division by zero.");
                alpha = numerator / denominator;
                x += alpha * d;
                j++;
            } while (j < 3 && alpha * alpha * deltaD > epsilon * epsilon);

            r = _problem->getGradient();
            r *= static_cast<data_t>(-1.0);

            const auto deltaOld = deltaNew;
            deltaNew = r.squaredL2Norm();

            if (deltaNew <= epsilon * epsilon * deltaZero) {
                // check that we are not stopping prematurely due to accumulated roundoff error
                r = _problem->getGradient();
                deltaNew = r.squaredL2Norm();
                if (deltaNew <= epsilon * epsilon * deltaZero) {
                    Logger::get("NLCG")->info("SUCCESS: Reached convergence at {}/{} iteration",
                                              i + 1, iterations);
                    return x;
                } else {
                    // we are very close to the desired solution, so do a hard reset
                    r *= static_cast<data_t>(-1.0);
                    d = r;
                }
            }

            data_t beta;
            switch (_beta) {
                case FR:
                    beta = deltaNew / deltaOld; // how to get transform?
                    break;
                case PR: {
                    auto deltaMid = r.dot(rOld);
                    beta = (deltaNew - deltaMid) / (deltaOld);
                    if (beta < 0)
                        beta = 0;
                } break;
            }

            d = r + beta * d;
        }

        return getCurrentSolution();
    }

    template <typename data_t>
    NLCG<data_t>* NLCG<data_t>::cloneImpl() const
    {
        return new NLCG<data_t>(*_problem, _beta);
    }

    template <typename data_t>
    bool NLCG<data_t>::isEqual(const Solver<data_t>& other) const
    {
        if (!Solver<data_t>::isEqual(other))
            return false;

        auto otherNLCG = dynamic_cast<const NLCG*>(&other);
        return otherNLCG;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class NLCG<float>;
    template class NLCG<double>;
} // namespace elsa