#pragma once

#include "LinearOperator.h"
#include "IdenticalBlocksDescriptor.h"
#include "Timer.h"

#include <limits>
#include <memory>

namespace elsa
{
    /**
     * @brief Operator representing a dictionary operation.
     *
     * @author Jonas Buerger - initial code
     *
     * @tparam data_t data type for the domain and range of the operator, defaulting to real_t
     *
     * This class represents a linear operator D that given a representation vector x
     * generates a signal y by multplication \f$ y = D*x \f$
     */
    template <typename data_t = real_t>
    class Dictionary : public LinearOperator<data_t>
    {
    public:
        /**
         * @brief Constructor for an empty dictionary.
         *
         * @param[in] signalDescriptor DataDescriptor describing the domain of the signals that
         * should be produced @param[in] nAtoms The number of atoms that should be in the dictionary
         */
        Dictionary(const DataDescriptor& signalDescriptor, index_t nAtoms);

        /**
         * @brief Constructor for an initialized dictionary.
         *
         * @param[in] dictionary DataContainer containing the entries of the dictionary
         * @throw InvalidArgumentError if dictionary doesn't have a IdenticalBlocksDescriptor or at
         * least one of the atoms is the 0-vector
         */
        explicit Dictionary(const DataContainer<data_t>& dictionary);

        /// default move constructor
        Dictionary(Dictionary<data_t>&& other) = default;

        /// default move assignment
        Dictionary& operator=(Dictionary<data_t>&& other) = default;

        /// default destructor
        ~Dictionary() override = default;

        /**
         * @brief Update a single atom of the dictionary with a new atom
         *
         * @param[in] j Index of the atom that should be updated
         * @param[in] atom DataContainer containing the new atom
         * @throw InvalidArgumentError if atom has the wrong size or index is out of bounds or atom
         * is the 0-vector
         */
        void updateAtom(index_t j, const DataContainer<data_t>& atom);

        void updateAtoms(const DataContainer<data_t>& atoms);

        /**
         * @brief Get an atom of the dictionary by index
         *
         * @param[in] j Index of the atom that should returned
         * @returns The atom in a DataContainer
         * @throw InvalidArgumentError if index is out of bounds
         */
        const DataContainer<data_t> getAtom(index_t j) const;

        /**
         * @brief Get the atoms of the dictionary, i.e. the underlying matrix
         *
         * @returns The atoms in a data container with IdenticalBlocksDescriptor where each block is
         * one atom
         */
        const DataContainer<data_t>& getAtoms() const;

        /**
         * @brief Returns the number of atoms in the dictionary
         *
         * @returns The number of atoms
         */
        index_t getNumberOfAtoms() const;

        /**
         * @brief Get a new dictionary restricted to a given support
         *
         * @param[in] support List of indices for the atoms that should be in the new dictionary
         * @returns A dictionary containing only the atoms that are defined by support
         * @throw InvalidArgumentError if support contains and index that is out of bounds
         */
        Dictionary<data_t> getSupportedDictionary(IndexVector_t support) const;

    protected:
        /// Copy constructor for internal usage
        Dictionary(const Dictionary<data_t>&) = default;

        /// apply the dictionary operation
        void applyImpl(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const override;

        /// apply the adjoint of the dictionary operation
        void applyAdjointImpl(const DataContainer<data_t>& y,
                              DataContainer<data_t>& Aty) const override;

        /// implement the polymorphic clone operation
        Dictionary<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override;

    private:
        /// the actual dictionary
        DataContainer<data_t> _dictionary;
        /// lift the base class variable for the range (signal) descriptor
        using LinearOperator<data_t>::_rangeDescriptor;
        /// the number of atoms in the dictionary
        index_t _nAtoms;

        static const IdenticalBlocksDescriptor&
            getIdenticalBlocksDescriptor(const DataContainer<data_t>& data);
        static DataContainer<data_t> generateInitialData(const DataDescriptor& signalDescriptor,
                                                         index_t nAtoms);
    };

} // namespace elsa
