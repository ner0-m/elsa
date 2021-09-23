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
        _representations = 1; // not the best initialization but OMP starts from scratch anyways
        calculateError();
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
    const Dictionary<data_t>& DictionaryLearningProblem<data_t>::getDictionary()
    {
        return _dictionary;
    }

    template <typename data_t>
    const DataContainer<data_t>& DictionaryLearningProblem<data_t>::getRepresentations()
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
        findAffectedSignals(atom);
        if (_currentAffectedSignals.size() < 1)
            throw LogicError(
                "DictionaryLearningProblem::getRestrictedError: atom doesn't affect any signals");
        IdenticalBlocksDescriptor errorDescriptor(_currentAffectedSignals.size(),
                                                  _signals.getBlock(0).getDataDescriptor());
        _currentModifiedError = std::make_unique<DataContainer<data_t>>(errorDescriptor);

        index_t i = 0;
        for (index_t idx : _currentAffectedSignals) {
            _currentModifiedError->getBlock(i) =
                _residual.getBlock(idx)
                + _dictionary.getAtom(atom) * _representations.getBlock(idx)[atom];
            ++i;
        }

        return *_currentModifiedError;
    }

    template <typename data_t>
    void DictionaryLearningProblem<data_t>::calculateError()
    {
        index_t nSignals = getIdenticalBlocksDescriptor(_signals).getNumberOfBlocks();
        for (index_t i = 0; i < nSignals; ++i) {
            _residual.getBlock(i) =
                _signals.getBlock(i) - _dictionary.apply(_representations.getBlock(i));
        }
    }

    template <typename data_t>
    void DictionaryLearningProblem<data_t>::updateRepresentations(
        const DataContainer<data_t>& representations)
    {
        if (_representations.getDataDescriptor() != representations.getDataDescriptor())
            throw InvalidArgumentError("DictionaryLearningProblem::updateRepresentations: can't "
                                       "update to representations with different descriptor");

        _representations = representations;
        calculateError();
    }

    template <typename data_t>
    void DictionaryLearningProblem<data_t>::updateAtom(index_t atomIdx,
                                                       const DataContainer<data_t>& atom,
                                                       const DataContainer<data_t>& representation)
    {
        // we should add some sanity checks on atom and representation here

        bool hasValidError = true;
        if (atomIdx != _currentAtomIdx) {
            // should not happen as ideally we update the atom
            // for which we previously calculated the error
            try {
                getRestrictedError(atomIdx);
                // no exception => previous atom affects signals, continue as usual
            } catch (LogicError&) {
                hasValidError = false;
                // previous atom wasn't used, using usual update strategy with restricted error not
                // possible => update to new atom and calculate global error
            }
        }
        _dictionary.updateAtom(atomIdx, atom);

        if (!hasValidError)
            findAffectedSignals(atomIdx);

        index_t i = 0;
        for (auto idx : _currentAffectedSignals) {
            _representations.getBlock(idx)[atomIdx] = representation[i];

            if (hasValidError) {
                // update relevant part of the error matrix
                _residual.getBlock(idx) =
                    _currentModifiedError->getBlock(i) - atom * representation[i];
            }
            ++i;
        }

        if (!hasValidError)
            calculateError();
    }

    template <typename data_t>
    void DictionaryLearningProblem<data_t>::findAffectedSignals(index_t atom)
    {
        _currentAffectedSignals = IndexVector_t(0);
        _currentAtomIdx = atom;
        index_t nSignals = getIdenticalBlocksDescriptor(_signals).getNumberOfBlocks();

        for (index_t i = 0; i < nSignals; ++i) {
            if (_representations.getBlock(i)[atom] != 0) {
                _currentAffectedSignals.conservativeResize(_currentAffectedSignals.size() + 1);
                _currentAffectedSignals[_currentAffectedSignals.size() - 1] = i;
            }
        }
    }

    // ------------------------------------------
    // explicit template instantiation
    template class DictionaryLearningProblem<float>;
    template class DictionaryLearningProblem<double>;

} // namespace elsa
