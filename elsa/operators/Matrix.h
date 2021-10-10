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
    class Matrix : public LinearOperator<data_t>
    {
    public:
        /**
         * @brief Constructor for an empty matrix.
         *
         * @param[in] signalDescriptor DataDescriptor describing the domain of the signals that
         * should be produced @param[in] nAtoms The number of atoms that should be in the dictionary
         */
        Matrix(const DataDescriptor& descriptor);

        /**
         * @brief Constructor for an initialized dictionary.
         *
         * @param[in] dictionary DataContainer containing the entries of the dictionary
         * @throw InvalidArgumentError if dictionary doesn't have a IdenticalBlocksDescriptor or at
         * least one of the atoms is the 0-vector
         */
        explicit Matrix(const DataContainer<data_t>& data);

        /// default move constructor
        Matrix(Matrix<data_t>&& other) = default;

        /// default move assignment
        Matrix& operator=(Matrix<data_t>&& other) = default;

        /// default destructor
        ~Matrix() override = default;

    protected:
        /// Copy constructor for internal usage
        Matrix(const Matrix<data_t>&) = default;

        /// apply the dictionary operation
        void applyImpl(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const override;

        /// apply the adjoint of the dictionary operation
        void applyAdjointImpl(const DataContainer<data_t>& y,
                              DataContainer<data_t>& Aty) const override;

        /// implement the polymorphic clone operation
        Matrix<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override;

    private:
        /// the actual data
        DataContainer<data_t> _matrix;

        /// lift the base class variable for the range and domain descriptors
        using LinearOperator<data_t>::_domainDescriptor;
        using LinearOperator<data_t>::_rangeDescriptor;

        static VolumeDescriptor initDomainDescriptor(const DataDescriptor& descriptor);
        static VolumeDescriptor initRangeDescriptor(const DataDescriptor& descriptor);
    };

} // namespace elsa
