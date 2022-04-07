#include "TikhonovProblem.h"
#include "L2NormPow2.h"
#include "WeightedL2NormPow2.h"

namespace elsa
{
    template <typename data_t>
    TikhonovProblem<data_t>::TikhonovProblem(const LinearOperator<data_t>& A,
                                             const DataContainer<data_t> b, real_t weight)
        : Problem<data_t>{
            L2NormPow2<data_t>(A, b),
            RegularizationTerm<data_t>(weight, L2NormPow2<data_t>(A.getDomainDescriptor()))}
    {
    }

    template <typename data_t>
    TikhonovProblem<data_t>::TikhonovProblem(
        const WLSProblem<data_t>& wlsProblem,
        const std::vector<RegularizationTerm<data_t>>& regTerms)
        : Problem<data_t>{wlsProblem.getDataTerm(), regTerms, wlsProblem.getCurrentSolution()}
    {
        // make sure that at least one regularization term exists
        if (regTerms.empty()) {
            throw InvalidArgumentError(
                "TikhonovProblem: at least one regularization term has to be supplied");
        }

        // make sure that all regularization terms are linear and of type (Weighted)L2NormPow2
        for (const auto& regTerm : regTerms) {
            const auto& func = regTerm.getFunctional();
            if (!is<L2NormPow2<data_t>>(func) && !is<WeightedL2NormPow2<data_t>>(func)) {
                throw InvalidArgumentError("TikhonovProblem: all regularization terms should be "
                                           "of type L2NormPow2 or WeightedL2NormPow2");
            }
            if (!is<LinearResidual<data_t>>(func.getResidual())) {
                throw InvalidArgumentError(
                    "TikhonovProblem: all regularization terms should be linear");
            }
        }
    }

    template <typename data_t>
    TikhonovProblem<data_t>::TikhonovProblem(const WLSProblem<data_t>& wlsProblem,
                                             const RegularizationTerm<data_t>& regTerm)
        : TikhonovProblem{wlsProblem, std::vector<RegularizationTerm<data_t>>{regTerm}}
    {
    }

    template <typename data_t>
    TikhonovProblem<data_t>* TikhonovProblem<data_t>::cloneImpl() const
    {
        return new TikhonovProblem(*this);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class TikhonovProblem<float>;
    template class TikhonovProblem<double>;
    template class TikhonovProblem<complex<float>>;
    template class TikhonovProblem<complex<double>>;

} // namespace elsa
