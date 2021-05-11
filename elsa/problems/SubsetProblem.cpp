#include "SubsetProblem.h"

namespace elsa
{

    template <typename data_t>
    SubsetProblem<data_t>::SubsetProblem(
        const Problem<data_t>& fullProblem,
        const std::vector<std::unique_ptr<Problem<data_t>>>& subsetProblems)
        : Problem<data_t>(fullProblem.getDataTerm(), fullProblem.getRegularizationTerms(),
                          fullProblem.getCurrentSolution()),
          _subsetProblems(0)
    {
        // TODO: maybe add a sanity check to make sure the domain of all problems matches
        for (const auto& problem : subsetProblems) {
            _subsetProblems.emplace_back(problem->clone());
        }
    }

    template <typename data_t>
    SubsetProblem<data_t>::SubsetProblem(const SubsetProblem<data_t>& subsetProblem)
        : Problem<data_t>(subsetProblem.getDataTerm(), subsetProblem.getRegularizationTerms(),
                          subsetProblem.getCurrentSolution()),
          _subsetProblems(0)
    {
        for (const auto& problem : subsetProblem._subsetProblems) {
            _subsetProblems.emplace_back(problem->clone());
        }
    }

    template <typename data_t>
    DataContainer<data_t> SubsetProblem<data_t>::getSubsetGradient(index_t subset)
    {
        if (subset < 0 || static_cast<std::size_t>(subset) >= _subsetProblems.size()) {
            throw std::invalid_argument(
                "SubsetProblem: subset index out of bounds for number of subsets");
        }

        _subsetProblems[static_cast<std::size_t>(subset)]->getCurrentSolution() =
            this->getCurrentSolution();
        return _subsetProblems[static_cast<std::size_t>(subset)]->getGradient();
    }

    template <typename data_t>
    void SubsetProblem<data_t>::getSubsetGradient(DataContainer<data_t>& result, index_t subset)
    {
        if (subset < 0 || static_cast<std::size_t>(subset) >= _subsetProblems.size()) {
            throw std::invalid_argument(
                "SubsetProblem: subset index out of bounds for number of subsets");
        }
        _subsetProblems[static_cast<std::size_t>(subset)]->getCurrentSolution() =
            this->getCurrentSolution();
        _subsetProblems[static_cast<std::size_t>(subset)]->getGradient(result);
    }

    template <typename data_t>
    index_t SubsetProblem<data_t>::getNumberOfSubsets() const
    {
        return static_cast<index_t>(_subsetProblems.size());
    }

    template <typename data_t>
    auto SubsetProblem<data_t>::cloneImpl() const -> SubsetProblem<data_t>*
    {
        return new SubsetProblem(*this);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class SubsetProblem<float>;
    template class SubsetProblem<double>;
} // namespace elsa
