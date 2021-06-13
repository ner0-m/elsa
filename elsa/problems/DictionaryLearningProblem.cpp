#include "DictionaryLearningProblem.h"

namespace elsa
{
    template <typename data_t>
    DictionaryLearningProblem<data_t>::DictionaryLearningProblem(
        const DataContainer<data_t>& signals, const index_t nAtoms)
        : _dictionary(signals.getBlock(0), nAtoms),
          _signals(signals),
          _residual(signals.getDataDescriptor())
    {
        VolumeDescriptor representationDescriptor({nAtoms});
        const auto& signalsDescriptor =
            dynamic_cast<const IdenticalBlocksDescriptor&>(_signals.getDataDescriptor());
        IdenticalBlocksDescriptor representationsDescriptor(signalsDescriptor.getNumberOfBlocks(),
                                                            representationDescriptor);
        _representations = DataContainer(representationsDescriptor);
        updateError();
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
    DataContainer<data_t> DictionaryLearningProblem<data_t>::getRestrictedError(index_t atom)
    {
        IndexVector_t affectedSignals(0);

        index_t i = 0;
        for (auto representation : _representations) {
            if (representation[atom] != 0) {
                affectedSignals.conservativeResize(affectedSignals.size() + 1);
                affectedSignals[affectedSignals.size() - 1] = i;
            }
            ++i;
        }

        IdenticalBlocksDescriptor errorDescriptor(affectedSignals.size(),
                                                  _signals.getBlock(0).getDataDescriptor());
        DataContainer<data_t> modifiedError(errorDescriptor);

        i = 0;
        for (index_t idx : affectedSignals) {
            modifiedError[i] =
                _residual[idx] + _dictionary.getAtom(atom) * _representations.getBlock(idx)[atom];
            ++i;
        }

        return modifiedError;
    }

    template <typename data_t>
    void DictionaryLearningProblem<data_t>::updateError()
    {
        index_t nSignals = _signals.getDataDescriptor().getNumberOfBlocks();
        for (index_t i; i < nSignals; ++i) {
            _residual.getBlock(i) =
                _signals.getBlock(i) - _dictionary.apply(_representations.getBlock(i));
        }
    }
} // namespace elsa
