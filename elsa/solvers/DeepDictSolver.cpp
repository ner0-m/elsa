#include "DeepDictSolver.h"

namespace elsa
{
    template <typename data_t>
    DeepDictSolver<data_t>::DeepDictSolver(DeepDictionaryLearningProblem<data_t>& problem,
                                           index_t sparsityLevel, data_t epsilon)
        : _problem{problem},
          //_nSamples{getNumberOfSamples(problem.getSignals())},
          _sparsityLevel{sparsityLevel},
          _epsilon{epsilon}
    {
    }

    template <typename data_t>
    const DeepDictionary<data_t>& DeepDictSolver<data_t>::getLearnedDeepDictionary()
    {
        return _problem.getDeepDictionary();
    }

    template <typename data_t>
    DataContainer<data_t> DeepDictSolver<data_t>::solve(index_t iterations)
    {

        /*
        Logger::get("DeepDictSolver")
            ->info("Started for {} iterations, with {} signals and {} atoms. "
                   "Stopping error: {}, initial error: {}",
                   iterations, _nSamples, dict.getNumberOfAtoms(), _epsilon,
                   _problem.getGlobalError().l2Norm());
        */
        index_t nDicts = _problem.getDeepDictionary().getNumberOfDictionaries();

        for (index_t level = 0; level < nDicts - 1; ++level) {
            index_t i = 0;
            while (i < iterations /*&& _problem.getGlobalError().l2Norm() >= _epsilon */) {
                DataContainer<data_t> representations(_problem.getRepresentationsDescriptor(level));
                index_t j = 0;
                for (auto p : _problem.getRepresentationWLSProblems(level)) {
                    CG<data_t> cg(p);
                    auto representation = cg.solve(1);
                    representations.getBlock(j) = representation;
                    ++j;
                }
                _problem.updateRepresentations(representations, level);

                DataContainer<data_t> transposedDictMatrix(
                    _problem.getTransposedDictDescriptor(level));
                j = 0;
                for (auto p : _problem.getDictionaryWLSProblems(level)) {
                    CG<data_t> cg(p);
                    auto representation = cg.solve(1);
                    for (const auto& x : representation) {
                        transposedDictMatrix(j) = x;
                        ++j;
                    }
                }
                _problem.updateDictionary(transposedDictMatrix, level);
            }
        }

        auto dictProblem = _problem.getDictionaryLearningProblem();
        // TODO expose params for sparsitylevel and niterations
        KSVD<data_t> ksvd(dictProblem, 10);
        auto representations = ksvd.solve(5);

        _problem.updateDictionary(ksvd.getLearnedDictionary(), nDicts - 1);
        _problem.updateRepresentations(representations, nDicts - 1);

        return representations;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class DeepDictSolver<float>;
    template class DeepDictSolver<double>;

} // namespace elsa
