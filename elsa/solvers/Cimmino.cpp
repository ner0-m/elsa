#include <BlockDescriptor.h>
#include "Cimmino.h"
#include "Logger.h"

namespace elsa
{
    template <typename data_t>
    Cimmino<data_t>::Cimmino(const WLSProblem<data_t>& problem, data_t relaxationParam)
        : Solver<data_t>(problem), _relaxationParam{relaxationParam}
    {
        auto linResid =
            dynamic_cast<const LinearResidual<data_t>*>(&(_problem->getDataTerm()).getResidual());

        if (!linResid)
            throw std::logic_error("Cimmino: Can only handle residuals of type 'LinearResidual'");
    }
    template <typename data_t>
    DataContainer<data_t>& Cimmino<data_t>::solveImpl(index_t iterations)
    {
        if (iterations == 0)
            iterations = _defaultIterations;
        auto& x = getCurrentSolution();
        auto linResid =
            static_cast<const LinearResidual<data_t>*>(&(_problem->getDataTerm()).getResidual());

        const auto& A = linResid->getOperator();
        const auto& b = linResid->getDataVector();
        const auto& rows = A.getRangeDescriptor().getNumberOfCoefficients();
        const auto& columns = A.getDomainDescriptor().getNumberOfCoefficients();
        DataContainer<data_t> DVec(A.getDomainDescriptor());

        for (index_t ii = 0; ii < columns; ii++) {
            DataContainer<data_t> vec(A.getDomainDescriptor());
            vec = 0;
            vec[ii] = 1;
            auto Avec = A.apply(vec);
            // 1 / l2 norm of the row of A
            DVec[ii] = 1 / A.applyAdjoint(Avec)[ii];
        }
        data_t m = static_cast<data_t>(1.) / static_cast<data_t>(rows);
        DVec *= m;

        for (index_t i = 0; i < iterations; ++i) {
            Logger::get("Cimmino")->info("iteration {} of {}", i + 1, iterations);
            auto d = Scaling(DVec.getDataDescriptor(), DVec);
            x += _relaxationParam * A.applyAdjoint(d.apply(b - A.apply(x)));
        }

        return getCurrentSolution();
    }

    template <typename data_t>
    Cimmino<data_t>* Cimmino<data_t>::cloneImpl() const
    {
        return new Cimmino<data_t>(*(static_cast<WLSProblem<data_t>*>(_problem.get())));
    }

    template <typename data_t>
    bool Cimmino<data_t>::isEqual(const Solver<data_t>& other) const
    {
        if (!Solver<data_t>::isEqual(other))
            return false;

        auto otherCimmino = dynamic_cast<const Cimmino*>(&other);
        return otherCimmino;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class Cimmino<float>;
    template class Cimmino<double>;
} // namespace elsa