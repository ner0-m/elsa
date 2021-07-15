#pragma once

#include "Dictionary.h"

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

        /// default destructor
        //~DictionaryLearningProblem() override = default;

        Dictionary<data_t>& getCurrentDictionary();

        DataContainer<data_t>& getCurrentRepresentations();

        DataContainer<data_t> getSignals();

        DataContainer<data_t> getGlobalError();

        DataContainer<data_t> getRestrictedError(IndexVector_t affectedSignals, index_t atom);

        void updateError();

        /*
            protected:
                /// implement the polymorphic clone operation
                RepresentationProblem<data_t>* cloneImpl() const override;

                /// override getGradient and throw exception if called
                void getGradientImpl(DataContainer<data_t>& result) override;
        */

    private:
        Dictionary<data_t> _dictionary;
        DataContainer<data_t> _representations;
        const DataContainer<data_t> _signals;
        DataContainer<data_t> _residual;

        static const IdenticalBlocksDescriptor&
            getIdenticalBlocksDescriptor(const DataContainer<data_t>& data);
    };
} // namespace elsa
