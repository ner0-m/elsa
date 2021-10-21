#include "DeepDictionaryLearningProblem.h"

namespace elsa
{
    template <typename data_t>
    DeepDictionaryLearningProblem<data_t>::DeepDictionaryLearningProblem(
        const DataContainer<data_t>& signals, std::vector<index_t> nAtoms,
        std::vector<ActivationFunction<data_t>> activationFunctions)
        : _deepDict(signals.getDataDescriptor(), nAtoms, activationFunctions), _signals(signals)
    {
        for (index_t i = 0; i < _deepDict.getNumberOfDictionaries(); ++i) {
            _representations.push_back(
                DataContainer<data_t>(_deepDict.getDictionary(i).getDomainDescriptor()));
        }
    }

    template <typename data_t>
    void DeepDictionaryLearningProblem<data_t>::updateDictionary(
        const DataContainer<data_t>& wlsSolution, index_t level)
    {
        auto dictMatrix = getTranspose(wlsSolution);
        _deepDict.getDictionary(level).updateAtoms(dictMatrix);
    }

    template <typename data_t>
    void DeepDictionaryLearningProblem<data_t>::updateDictionary(
        const Dictionary<data_t>& dictSolution, index_t level)
    {
        _deepDict.getDictionary(level).updateAtoms(dictSolution.getAtoms());
    }

    template <typename data_t>
    void DeepDictionaryLearningProblem<data_t>::updateRepresentations(
        const DataContainer<data_t>& wlsSolution, index_t level)
    {
        _representations.at(level) = wlsSolution;
    }

    template <typename data_t>
    std::vector<WLSProblem<data_t>>
        DeepDictionaryLearningProblem<data_t>::getDictionaryWLSProblems(index_t level)
    {
        if (level < 0 || level >= _deepDict.getNumberOfDictionaries()) {
            throw InvalidArgumentError("foo");
        }

        DataContainer<data_t> signals = (level == 0 ? _signals : _representations.at(level - 1));
        if (level > 0)
            std::for_each(signals.begin(), signals.end(),
                          _deepDict.getActivationFunction(level).getInverse());

        std::vector<WLSProblem<data_t>> problems;

        auto representation = _representations.at(level);
        const auto& representationDescriptor =
            downcast_safe<IdenticalBlocksDescriptor>(representation.getDataDescriptor());

        auto representation_T = getTranspose(representation.viewAs(VolumeDescriptor{
            representationDescriptor.getDescriptorOfBlock(0).getNumberOfCoefficients(),
            representationDescriptor.getNumberOfBlocks()}));

        Matrix<data_t> matOp(representation_T);

        const auto& signalDescriptor =
            downcast_safe<IdenticalBlocksDescriptor>(signals.getDataDescriptor());

        for (index_t i = 0; i < signalDescriptor.getNumberOfBlocks(); ++i) {
            WLSProblem problem(matOp, signals.getBlock(i));
            problems.push_back(problem);
        }

        return problems;
    }

    template <typename data_t>
    std::vector<WLSProblem<data_t>>
        DeepDictionaryLearningProblem<data_t>::getRepresentationWLSProblems(index_t level)
    {
        std::vector<WLSProblem<data_t>> problems;

        if (level < 0 || level >= _deepDict.getNumberOfDictionaries()) {
            throw InvalidArgumentError("foo");
        }

        DataContainer<data_t> signals = (level == 0 ? _signals : _representations.at(level - 1));
        if (level > 0)
            std::for_each(signals.begin(), signals.end(),
                          _deepDict.getActivationFunction(level).getInverse());

        const auto& dict = _deepDict.getDictionary(level);

        auto representation = _representations.at(level);
        const auto& signalDescriptor =
            downcast_safe<IdenticalBlocksDescriptor>(signals.getDataDescriptor());

        for (index_t i = 0; i < signalDescriptor.getNumberOfBlocks(); ++i) {
            WLSProblem problem(dict, signals.getBlock(i));
            problems.push_back(problem);
        }

        return problems;
    }

    template <typename data_t>
    DictionaryLearningProblem<data_t>
        DeepDictionaryLearningProblem<data_t>::getDictionaryLearningProblem()
    {
        auto lastIdx = _deepDict.getNumberOfDictionaries() - 1;
        return DictionaryLearningProblem(_representations.at(lastIdx),
                                         _deepDict.getDictionary(lastIdx).getNumberOfAtoms());
    }

    template <typename data_t>
    const DeepDictionary<data_t>& DeepDictionaryLearningProblem<data_t>::getDeepDictionary()
    {
        return _deepDict;
    }

    template <typename data_t>
    const DataDescriptor&
        DeepDictionaryLearningProblem<data_t>::getRepresentationsDescriptor(index_t level)
    {
        return _representations.at(level).getDataDescriptor();
    }

    template <typename data_t>
    VolumeDescriptor
        DeepDictionaryLearningProblem<data_t>::getTransposedDictDescriptor(index_t level)
    {
        const auto& dictDescriptor = downcast_safe<IdenticalBlocksDescriptor>(
            _deepDict.getDictionary(level).getAtoms().getDataDescriptor());
        return VolumeDescriptor({dictDescriptor.getNumberOfBlocks(),
                                 dictDescriptor.getDescriptorOfBlock(0).getNumberOfCoefficients()});
    }

    template <typename data_t>
    DataContainer<data_t>
        DeepDictionaryLearningProblem<data_t>::getTranspose(const DataContainer<data_t>& matrix)
    {
        // TODO sanity check
        auto nCoeffsPerDim = matrix.getDataDescriptor().getNumberOfCoefficientsPerDimension();
        DataContainer<data_t> matrix_T(VolumeDescriptor({nCoeffsPerDim[1], nCoeffsPerDim[0]}));

        for (index_t i = 0; i < nCoeffsPerDim[0]; ++i) {
            for (index_t j = 0; j < nCoeffsPerDim[1]; ++j) {
                matrix_T(j, i) = matrix(i, j);
            }
        }

        return matrix_T;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class DeepDictionaryLearningProblem<float>;
    template class DeepDictionaryLearningProblem<double>;

} // namespace elsa
