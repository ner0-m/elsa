#include "WLSSubsetProblem.h"

namespace elsa
{

    template <typename data_t>
    WLSSubsetProblem<data_t>::WLSSubsetProblem(
        const LinearOperator<data_t>& A, const DataContainer<data_t>& b,
        const std::vector<std::unique_ptr<LinearOperator<data_t>>>& subsetAs)
        : SubsetProblem<data_t>(WLSProblem<data_t>(A, b), *wlsProblemsFromOperators(subsetAs, b))
    {
    }

    template <typename data_t>
    auto WLSSubsetProblem<data_t>::cloneImpl() const -> WLSSubsetProblem<data_t>*
    {
        return new WLSSubsetProblem<data_t>(*this);
    }

    template <typename data_t>
    std::unique_ptr<std::vector<std::unique_ptr<Problem<data_t>>>>
        WLSSubsetProblem<data_t>::wlsProblemsFromOperators(
            const std::vector<std::unique_ptr<LinearOperator<data_t>>>& subsetAs,
            const DataContainer<data_t>& b)
    {
        // Checks for blocked data descriptor and block index out of bounds are already done in
        // the DataContainer, no need to do them here.

        auto subProblems = std::make_unique<std::vector<std::unique_ptr<Problem<data_t>>>>(0);
        for (long unsigned int i = 0; i < subsetAs.size(); i++) {
            // materialize block, as else this can be a dangling pointer and we do not want to store
            // the view
            subProblems->emplace_back(std::make_unique<WLSProblem<data_t>>(
                *subsetAs[i], materialize(b.getBlock(static_cast<index_t>(i)))));
        }

        return subProblems;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class WLSSubsetProblem<float>;
    template class WLSSubsetProblem<double>;
} // namespace elsa
