#pragma once

#include "Dictionary.h"
#include <memory>

namespace elsa
{
    /**
     * @brief Class representing a dictionary learning problem.
     *
     * @author Jonas Buerger - initial code
     *
     * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
     *
     * TODO: add description
     */
    template <typename data_t = real_t>
    class DictionaryLearningProblem
    {
    public:
        /**
         * @brief Constructor for the dictionary learning problem, accepting signals and number of
         * atoms
         *
         * @param[in] signals The signals for which a dictionary with sparse representations should
         * be found
         * @param[in] nAtoms The number of atoms the learned dictionary will have
         */
        DictionaryLearningProblem(const DataContainer<data_t>& signals, index_t nAtoms);

        const Dictionary<data_t>& getDictionary();

        const DataContainer<data_t>& getRepresentations();

        DataContainer<data_t> getSignals();

        DataContainer<data_t> getGlobalError();

        DataContainer<data_t> getRestrictedError(index_t atom);

        void updateRepresentations(const DataContainer<data_t>& representations);

        void updateAtom(index_t atomIdx, const DataContainer<data_t>& atom,
                        const DataContainer<data_t>& representation);

    private:
        Dictionary<data_t> _dictionary;
        DataContainer<data_t> _representations;
        const DataContainer<data_t> _signals;
        DataContainer<data_t> _residual;

        IndexVector_t _currentAffectedSignals;
        index_t _currentAtomIdx;
        std::unique_ptr<DataContainer<data_t>> _currentModifiedError;

        void calculateError();
        void findAffectedSignals(index_t atom);
        static const IdenticalBlocksDescriptor&
            getIdenticalBlocksDescriptor(const DataContainer<data_t>& data);
    };
} // namespace elsa
