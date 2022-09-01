#include "SubsetProblem.h"
#include "TypeCasts.hpp"

namespace elsa
{

    template <typename data_t>
    SubsetProblem<data_t>::SubsetProblem(
        const Problem<data_t>& fullProblem,
        const std::vector<std::unique_ptr<Problem<data_t>>>& subsetProblems)
        : Problem<data_t>(fullProblem.getDataTerm(), fullProblem.getRegularizationTerms()),

          _subsetProblems(0)
    {
        // TODO: maybe add a sanity check to make sure the domain of all problems matches
        for (const auto& problem : subsetProblems) {
            _subsetProblems.emplace_back(problem->clone());
        }
    }

    template <typename data_t>
    SubsetProblem<data_t>::SubsetProblem(const SubsetProblem<data_t>& subsetProblem)
        : Problem<data_t>(subsetProblem.getDataTerm(), subsetProblem.getRegularizationTerms()),
          _subsetProblems(0)
    {
        for (const auto& problem : subsetProblem._subsetProblems) {
            _subsetProblems.emplace_back(problem->clone());
        }
    }

    template <typename data_t>
    DataContainer<data_t> SubsetProblem<data_t>::getSubsetGradient(const DataContainer<data_t>& x,
                                                                   index_t subset)
    {
        if (subset < 0 || asUnsigned(subset) >= _subsetProblems.size()) {
            throw std::invalid_argument(
                "SubsetProblem: subset index out of bounds for number of subsets");
        }

        return _subsetProblems[asUnsigned(subset)]->getGradient(x);
    }

    template <typename data_t>
    void SubsetProblem<data_t>::getSubsetGradient(const DataContainer<data_t>& x,
                                                  DataContainer<data_t>& result, index_t subset)
    {
        if (subset < 0 || asUnsigned(subset) >= _subsetProblems.size()) {
            throw std::invalid_argument(
                "SubsetProblem: subset index out of bounds for number of subsets");
        }
        _subsetProblems[asUnsigned(subset)]->getGradient(x, result);
    }

    template <typename data_t>
    index_t SubsetProblem<data_t>::getNumberOfSubsets() const
    {
        return asSigned(_subsetProblems.size());
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
