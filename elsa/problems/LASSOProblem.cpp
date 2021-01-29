#include "LASSOProblem.h"
#include "Identity.h"

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
            throw std::invalid_argument(
                "LASSOProblem: regularization term must have a non-negative weight");
        }
        if (!dynamic_cast<const L1Norm<data_t>*>(&regTerm.getFunctional())) {
            throw std::invalid_argument("LASSOProblem: regularization term must be type L1Norm");
        }
    }

    template <typename data_t>
    LASSOProblem<data_t>::LASSOProblem(const Problem<data_t>& problem)
        : LASSOProblem<data_t>{wlsFromProblem(problem), regTermFromProblem(problem)}
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

    template <typename data_t>
    auto LASSOProblem<data_t>::wlsFromProblem(const Problem<data_t>& problem) -> WLSProblem<data_t>
    {
        auto linResid =
            dynamic_cast<const LinearResidual<data_t>*>(&(problem.getDataTerm()).getResidual());

        std::unique_ptr<LinearOperator<data_t>> dataTermOp;

        if (linResid->hasOperator()) {
            dataTermOp = linResid->getOperator().clone();
        } else {
            dataTermOp = std::make_unique<Identity<data_t>>(linResid->getDomainDescriptor());
        }

        DataContainer<data_t> dataVec = [&] {
            if (linResid->hasDataVector()) {
                return DataContainer<data_t>(linResid->getDataVector());
            } else {
                Eigen::Matrix<data_t, Eigen::Dynamic, 1> zeroes(
                    linResid->getRangeDescriptor().getNumberOfCoefficients());
                zeroes.setZero();

                return DataContainer<data_t>(linResid->getRangeDescriptor(), zeroes);
            }
        }();

        return WLSProblem<data_t>(*dataTermOp, dataVec);
    }

    template <typename data_t>
    auto LASSOProblem<data_t>::regTermFromProblem(const Problem<data_t>& problem)
        -> RegularizationTerm<data_t>
    {
        const auto& regTerms = problem.getRegularizationTerms();

        if (regTerms.size() != 1) {
            throw std::invalid_argument(
                "LASSOProblem: exactly one regularization term is required");
        }

        return regTerms[0];
    }

    // ------------------------------------------
    // explicit template instantiation
    template class LASSOProblem<float>;
    template class LASSOProblem<double>;
} // namespace elsa
