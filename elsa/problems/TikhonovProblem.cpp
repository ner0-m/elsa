#include "TikhonovProblem.h"
#include "L2NormPow2.h"
#include "WeightedL2NormPow2.h"

namespace elsa
{
    template <typename data_t>
    TikhonovProblem<data_t>::TikhonovProblem(
        const WLSProblem<data_t>& wlsProblem,
        const std::vector<RegularizationTerm<data_t>>& regTerms)
        : Problem<data_t>{wlsProblem.getDataTerm(), regTerms, wlsProblem.getCurrentSolution()}
    {
        // make sure that at least one regularization term exists
        if (regTerms.empty()) {
            throw std::invalid_argument(
                "TikhonovProblem: at least one regularization term has to be supplied");
        }

        // make sure that all regularization terms are linear and of type (Weighted)L2NormPow2
        for (const auto& regTerm : regTerms) {
            const auto& func = regTerm.getFunctional();
            if (!dynamic_cast<const L2NormPow2<data_t>*>(&func)
                && !dynamic_cast<const WeightedL2NormPow2<data_t>*>(&func)) {
                throw std::invalid_argument("TikhonovProblem: all regularization terms should be "
                                            "of type L2NormPow2 or WeightedL2NormPow2");
            }
            if (!dynamic_cast<const LinearResidual<data_t>*>(&func.getResidual())) {
                throw std::invalid_argument(
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
    template class TikhonovProblem<std::complex<float>>;
    template class TikhonovProblem<std::complex<double>>;

} // namespace elsa