#include "DictionaryLearningProblem.h"

namespace elsa
{
    template <typename data_t>
    DictionaryLearningProblem<data_t>::DictionaryLearningProblem(
        const DataContainer<data_t>& signals, const index_t nAtoms)
        : _dictionary(getIdenticalBlocksDescriptor(signals).getDescriptorOfBlock(0), nAtoms),
          _signals(signals),
          _residual(signals.getDataDescriptor()),
          _representations(
              IdenticalBlocksDescriptor(getIdenticalBlocksDescriptor(signals).getNumberOfBlocks(),
                                        VolumeDescriptor({nAtoms})))
    {
        updateError();
    }

    template <typename data_t>
    const IdenticalBlocksDescriptor&
        DictionaryLearningProblem<data_t>::getIdenticalBlocksDescriptor(
            const DataContainer<data_t>& data)
    {
        try {
            const auto& identBlocksDesc =
                dynamic_cast<const IdenticalBlocksDescriptor&>(data.getDataDescriptor());
            return identBlocksDesc;
        } catch (std::bad_cast e) {
            throw InvalidArgumentError("DictionaryLearningProblem: cannot initialize from signals "
                                       "without IdenticalBlocksDescriptor");
        }
    }

    template <typename data_t>
    Dictionary<data_t>& DictionaryLearningProblem<data_t>::getCurrentDictionary()
    {
        return _dictionary;
    }

    template <typename data_t>
    DataContainer<data_t>& DictionaryLearningProblem<data_t>::getCurrentRepresentations()
    {
        return _representations;
    }

    template <typename data_t>
    DataContainer<data_t> DictionaryLearningProblem<data_t>::getSignals()
    {
        return _signals;
    }

    template <typename data_t>
    DataContainer<data_t> DictionaryLearningProblem<data_t>::getGlobalError()
    {
        return _residual;
    }

    template <typename data_t>
    DataContainer<data_t>
        DictionaryLearningProblem<data_t>::getRestrictedError(IndexVector_t affectedSignals,
                                                              index_t atom)
    {
        IdenticalBlocksDescriptor errorDescriptor(affectedSignals.size(),
                                                  _signals.getBlock(0).getDataDescriptor());
        DataContainer<data_t> modifiedError(errorDescriptor);

        index_t i = 0;
        for (index_t idx : affectedSignals) {
            modifiedError.getBlock(i) =
                _residual.getBlock(idx)
                + _dictionary.getAtom(atom) * _representations.getBlock(idx)[atom];
            ++i;
        }

        return modifiedError;
    }

    template <typename data_t>
    void DictionaryLearningProblem<data_t>::updateError()
    {
        index_t nSignals = getIdenticalBlocksDescriptor(_signals).getNumberOfBlocks();
        for (index_t i = 0; i < nSignals; ++i) {
            _residual.getBlock(i) =
                _signals.getBlock(i) - _dictionary.apply(_representations.getBlock(i));
        }
    }

    // ------------------------------------------
    // explicit template instantiation
    template class DictionaryLearningProblem<float>;
    template class DictionaryLearningProblem<double>;

} // namespace elsa
