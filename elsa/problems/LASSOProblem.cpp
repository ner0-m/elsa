#include "LASSOProblem.h"
#include "Error.h"
#include "Identity.h"
#include "Into.hpp"
#include "TypeCasts.hpp"

namespace elsa
{
    template <typename data_t>
    LASSOProblem<data_t>::LASSOProblem(WLSProblem<data_t> wlsProblem,
                                       const RegularizationTerm<data_t>& regTerm)
        : Problem<data_t>{wlsProblem.getDataTerm(),
                          std::vector<RegularizationTerm<data_t>>{regTerm},
                          wlsProblem.getCurrentSolution()},
          _wlsProblem{wlsProblem}
    {
        if (regTerm.getWeight() < 0) {
            throw InvalidArgumentError(
                "LASSOProblem: regularization term must have a non-negative weight");
        }
        if (!is<L1Norm<data_t>>(regTerm.getFunctional())) {
            throw InvalidArgumentError("LASSOProblem: regularization term must be type L1Norm");
        }
    }

    template <typename data_t>
    LASSOProblem<data_t>::LASSOProblem(Into<LASSOProblem<data_t>> into)
        : LASSOProblem<data_t>(
            into.into()._wlsProblem,
            into.into().getRegularizationTerms()[0]) // Save, as else conversion would fail
    {
    }

    template <typename data_t>
    auto LASSOProblem<data_t>::cloneImpl() const -> LASSOProblem<data_t>*
    {
        return new LASSOProblem<data_t>(*this);
    }

    template <typename data_t>
    auto LASSOProblem<data_t>::getLipschitzConstantImpl(index_t nIterations) const -> data_t
    {
        // compute the Lipschitz Constant of the WLSProblem as the reg. term is not differentiable
        return _wlsProblem.getLipschitzConstant(nIterations);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class LASSOProblem<float>;
    template class LASSOProblem<double>;
    template class LASSOProblem<std::complex<float>>;
    template class LASSOProblem<std::complex<double>>;
} // namespace elsa
