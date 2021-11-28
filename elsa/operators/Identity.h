#pragma once

#include "LinearOperator.h"

namespace elsa
{
    /**
     * @brief Operator representing the identity operation.
     *
     * @author Matthias Wieczorek - initial code
     * @author Tobias Lasser - modularization, rewrite
     *
     * @tparam data_t data type for the domain and range of the operator, defaulting to real_t
     *
     * This class represents a linear operator A that is the identity, i.e. Ax = x.
     */
    template <typename data_t = real_t>
    class Identity : public LinearOperator<data_t>
    {
    public:
        /**
         * @brief Constructor for the identity operator, specifying the domain (= range).
         *
         * @param[in] descriptor DataDescriptor describing the domain and range of the operator
         */
        explicit Identity(const DataDescriptor& descriptor);

        /**
         * @brief Constructor for the identity operator, specifying the domain and range.
         *
         * @param[in] domainDescriptor DataDescriptor describing the domain of the operator
         * @param[in] rangeDescriptor DataDescriptor describing the range of the operator
         */
        Identity(const DataDescriptor& domainDescriptor, const DataDescriptor& rangeDescriptor);

        /// default destructor
        ~Identity() override = default;

    protected:
        /// default copy constructor, hidden from non-derived classes to prevent potential slicing
        Identity(const Identity<data_t>&) = default;

        /**
         * @brief apply the identity operator A to x, i.e. Ax = x
         *
         * @param[in] x input DataContainer (in the domain of the operator)
         * @param[out] Ax output DataContainer (in the range of the operator)
         */
        void applyImpl(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const override;

        /**
         * @brief apply the adjoint of the identity operator A to y, i.e. A^ty = y
         *
         * @param[in] y input DataContainer (in the range of the operator)
         * @param[out] Aty output DataContainer (in the domain of the operator)
         */
        void applyAdjointImpl(const DataContainer<data_t>& y,
                              DataContainer<data_t>& Aty) const override;

        /// implement the polymorphic clone operation
        Identity<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override;
    };
} // namespace elsa
