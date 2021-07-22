#include "SIRT.h"
#include "Logger.h"
namespace elsa
{
    template <typename data_t>
    SIRT<data_t>::SIRT(const WLSProblem<data_t>& problem) : Solver<data_t>(problem)
    {
        auto linResid =
            dynamic_cast<const LinearResidual<data_t>*>(&(_problem->getDataTerm()).getResidual());

        if (!linResid)
            throw std::logic_error("SIRT: Can only handle residuals of type 'LinearResidual'");
    }
    template <typename data_t>
    auto SIRT<data_t>::solveImpl(index_t iterations) -> DataContainer<data_t>&
    {
        if (iterations == 0)
            iterations = _defaultIterations;

        const auto& linResid =
            downcast<LinearResidual<data_t>>((_problem->getDataTerm()).getResidual());

        auto& x = getCurrentSolution();
        const auto& A = linResid.getOperator();

        DataContainer<data_t> ones(A.getRangeDescriptor());
        ones = 1;
        DataContainer<data_t> c = A.applyAdjoint(ones);
        auto C = Scaling(c.getDataDescriptor(), c);

        for (index_t i = 0; i < iterations; ++i) {
            Logger::get("SIRT")->info("iteration {} of {}", i + 1, iterations);
            auto gradient = _problem->getGradient();
            x -= C.apply(gradient);
        }

        return getCurrentSolution();
    }

    template <typename data_t>
    SIRT<data_t>* SIRT<data_t>::cloneImpl() const
    {
        return new SIRT<data_t>(*(static_cast<WLSProblem<data_t>*>(_problem.get())));
    }

    template <typename data_t>
    bool SIRT<data_t>::isEqual(const Solver<data_t>& other) const
    {
        if (!Solver<data_t>::isEqual(other))
            return false;

        auto otherSIRT = dynamic_cast<const SIRT*>(&other);
        return static_cast<bool>(otherSIRT);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class SIRT<float>;
    template class SIRT<double>;
} // namespace elsa
