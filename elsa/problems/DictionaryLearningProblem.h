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
     * Note that this class does not derive from the problem class.
     * This class represents a dictionary learning problem, i.e.
     * \f$ \min_{D,x} \| Y - DX \|_2^2 < \epsilon \f$, where \f$ D \f$ is the
     * dictionary operator that should be learned for the signals \f$ Y \f$ with sparse
     * representations \f$ X \f$. The sparsity condition \f$ \min_x \|\| x \|\|_0 \f$ for the
     * columns \f$ x \f$ from \f$ X \f$ is not enforced by the class but handled implicitly, either
     * by using a greedy algorithm that starts with the 0-vector as a representation or by creating
     * another problem that explicitly takes sparsity into account and relaxes the L0-Norm to the
     * L1-Norm.
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
         * be found, must have an IdenticalBlocksDescriptor with each block being one signal.
         * @param[in] nAtoms The number of atoms the learned dictionary will have
         * @throw InvalidArgumentError if signals doesn't have a IdenticalBlocksDescriptor
         */
        DictionaryLearningProblem(const DataContainer<data_t>& signals, index_t nAtoms);

        /**
         * @brief Getter for the current dictionary
         * atoms
         *
         * @returns The dictionary
         */
        const Dictionary<data_t>& getDictionary();

        /**
         * @brief Getter for the current representations
         * atoms
         *
         * @returns A data container with IdenticalBlocksDescriptor, where each block is one
         * representation.
         */
        const DataContainer<data_t>& getRepresentations();

        /**
         * @brief Getter for the signals used in the problem
         * atoms
         *
         * @returns A data container with IdenticalBlocksDescriptor, where each block is one signal.
         */
        DataContainer<data_t> getSignals();

        /**
         * @brief Returns the current global error, i.e. \f$ \| Y - DX \|_2^2 \f$
         *
         * @returns A data container with IdenticalBlocksDescriptor, where each block corresponds to
         * the error of one signal.
         */
        DataContainer<data_t> getGlobalError();

        /**
         * @brief Returns the error with respect to a single atom, i.e., only signals that currently
         * use this atom are considered and the influence of other atoms on said signals is
         * neglected.
         *
         * @param[in] atom The index of the atom
         *
         * @returns The error restricted to atom
         */
        std::optional<DataContainer<data_t>> getRestrictedError(index_t atom);

        /**
         * @brief Update all representations \f$ X \f$
         *
         * @param[in] representations DataContainer with IdenticalBlocksDescriptor, where each block
         * corresponds to one representation
         *
         * @throw InvalidArgumentError if representations have a different DataDescriptor than the
         * one implied by nAtoms and the number of signals
         */
        void updateRepresentations(const DataContainer<data_t>& representations);

        /**
         * @brief Update a single atom in the dictionary together with all representations that use
         * this atom. Should be called for the same atom for which getRestrictedError() has
         * previously been called.
         *
         * @param[in] atomIdx The index of the atom that should be updated
         * @param[in] atom The new atom
         * @param[in] representation A vector where each entry contains the new value for a
         * representation that uses the atom. Note that this is not a single complete
         * representation, speaking in terms of the matrix of representations \f$ X \f$ it is a
         * (partial) row, not a column.
         *
         */
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
