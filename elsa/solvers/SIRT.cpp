//
// Created by Maryna on 4/12/2021.
//

#include "SIRT.h"
#include "Logger.h"

namespace elsa
{
    template <typename data_t>
    SIRT<data_t>::SIRT(const WLSProblem<data_t>& problem) : Solver<data_t>(problem)
    {
    }
    template <typename data_t>
    DataContainer<data_t>& SIRT<data_t>::solveImpl(index_t iterations)
    {
        if (iterations == 0)
            iterations = _defaultIterations;

        for (index_t i = 0; i < iterations; ++i) {
            Logger::get("SIRT")->info("iteration {} of {}", i + 1, iterations);
            auto& x = getCurrentSolution();
            auto gradient = _problem->getGradient();
            auto linResid = dynamic_cast<const LinearResidual<data_t>*>(
                &(_problem->getDataTerm()).getResidual());

            if (!linResid)
                throw std::logic_error("SIRT: Incorrect type of residual");

            const LinearOperator<data_t>& A = linResid->getOperator();

            DataContainer<data_t> dc1(A.getRangeDescriptor());
            dc1 = 1; // vector of 1s
            DataContainer<data_t> c = A.applyAdjoint(dc1);
            auto C = Scaling(c.getDataDescriptor(), c); // which DataDescriptor?

            x -= C.apply(gradient);
        }

        return getCurrentSolution();
    }
    template <typename data_t>
    auto SIRT<data_t>::cloneImpl() const -> SIRT<data_t>*
    {
        return new SIRT(*_problem);
    }

    template <typename data_t>
    auto SIRT<data_t>::isEqual(const Solver<data_t>& other) const -> bool
    {
        if (!Solver<data_t>::isEqual(other))
            return false;

        auto otherSIRT = dynamic_cast<const SIRT*>(&other);
        return otherSIRT;
    }
} // namespace elsa
