#pragma once

#include "Dictionary.h"
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
    class DeepDictionary : public LinearOperator<data_t>
    {
    public:
        /**
         * @brief Constructor for an empty dictionary.
         *
         * @param[in] signalDescriptor DataDescriptor describing the domain of the signals that
         * should be produced @param[in] nAtoms The number of atoms that should be in the dictionary
         */
        DeepDictionary(const DataDescriptor& signalDescriptor, const std::vector<index_t>& nAtoms,
                       const std::vector<std::function<data_t(data_t)>>& activationFunctions);

        /**
         * @brief Constructor for an initialized dictionary.
         *
         * @param[in] dictionary DataContainer containing the entries of the dictionary
         * @throw InvalidArgumentError if dictionary doesn't have a IdenticalBlocksDescriptor or at
         * least one of the atoms is the 0-vector
         */
        explicit DeepDictionary(
            const std::vector<Dictionary<data_t>>& dictionaries,
            const std::vector<std::function<data_t(data_t)>>& activationFunctions);

        /// default move constructor
        DeepDictionary(DeepDictionary<data_t>&& other) = default;

        /// default move assignment
        DeepDictionary& operator=(DeepDictionary<data_t>&& other) = default;

        /// default destructor
        ~DeepDictionary() override = default;

    protected:
        /// Copy constructor for internal usage
        DeepDictionary(const DeepDictionary<data_t>&) = default;

        /// apply the dictionary operation
        void applyImpl(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const override;

        /// apply the adjoint of the dictionary operation
        void applyAdjointImpl(const DataContainer<data_t>& y,
                              DataContainer<data_t>& Aty) const override;

        /// implement the polymorphic clone operation
        DeepDictionary<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override;

    private:
        /// the number of atoms per dictionary
        std::vector<index_t> _nAtoms;

        /// the activation functions, applied element wise after each layer
        std::vector<std::function<data_t(data_t)>> _activationFunctions;

        /// the dictionary operators
        std::vector<Dictionary<data_t>> _dictionaries;

        /// lift the base class variable for the range (signal) descriptor
        using LinearOperator<data_t>::_rangeDescriptor;
        using LinearOperator<data_t>::_domainDescriptor;

        static std::vector<Dictionary<data_t>> generateInitialData(
            const DataDescriptor& signalDescriptor, const std::vector<index_t>& nAtoms,
            const std::vector<std::function<data_t(data_t)>>& activationFunctions);
    };

} // namespace elsa
