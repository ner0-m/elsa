#pragma once

#include "DeepDictionary.h"
#include "DictionaryLearningProblem.h"
#include "Matrix.h"
#include "WLSProblem.h"
#include <memory>

namespace elsa
{
    /**
     * @brief Class representing a deep dictionary learning problem.
     *
     * @author Jonas Buerger - initial code
     *
     * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
     *
     * TODO: add description
     */
    template <typename data_t = real_t>
    class DeepDictionaryLearningProblem
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
        DeepDictionaryLearningProblem(const DataContainer<data_t>& signals,
                                      std::vector<index_t> nAtoms,
                                      std::vector<ActivationFunction<data_t>> activationFunctions);

        DeepDictionaryLearningProblem(const DataContainer<data_t>& signals,
                                      std::vector<index_t> nAtoms);

        std::vector<WLSProblem<data_t>> getDictionaryWLSProblems(index_t level);

        std::vector<WLSProblem<data_t>> getRepresentationWLSProblems(index_t level);

        DictionaryLearningProblem<data_t> getDictionaryLearningProblem();

        void updateDictionary(const DataContainer<data_t>& wlsSolution, index_t level);

        void updateDictionary(const Dictionary<data_t>& dictSolution, index_t level);

        void updateRepresentations(const DataContainer<data_t>& wlsSolution, index_t level);

        const DeepDictionary<data_t>& getDeepDictionary();

        const DataDescriptor& getRepresentationsDescriptor(index_t level);

        VolumeDescriptor getTransposedDictDescriptor(index_t level);
        /*
                const DataContainer<data_t>& getRepresentations();

                DataContainer<data_t> getSignals();

                DataContainer<data_t> getGlobalError();

                DataContainer<data_t> getRestrictedError(index_t atom);

                void updateRepresentations(const DataContainer<data_t>& representations);

                void updateAtom(index_t atomIdx, const DataContainer<data_t>& atom,
                                const DataContainer<data_t>& representation);

        */
    private:
        DeepDictionary<data_t> _deepDict;

        const DataContainer<data_t> _signals;

        std::vector<DataContainer<data_t>> _representations;

        static DataContainer<data_t> getTranspose(const DataContainer<data_t>& matrix);

        static const IdenticalBlocksDescriptor&
            getIdenticalBlocksDescriptor(const DataContainer<data_t>& data);
    };
} // namespace elsa
