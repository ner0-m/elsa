#pragma once

#include "LinearOperator.h"

namespace elsa
{
    /**
     * @brief Operator representing a zero operation.
     *
     * @author Shen Hu - initial code
     *
     * @tparam data_t data type for the domain and range of the operator, defaulting to real_t
     *
     * This class represents an operator A that maps any vector in the domain into
     * a zero vector in the range, i.e. Ax = 0.
     */
    template <typename data_t = real_t>
    class ZeroOperator : public LinearOperator<data_t>
    {
    public:
        /**
         * @brief Constructor for ZeroOperator, specifying the domain and range.
         *
         * @param[in] domainDescriptor DataDescriptor describing the domain of the operator
         * @param[in] rangeDescriptor DataDescriptor describing the range of the operator
         */
        ZeroOperator(const DataDescriptor& domainDescriptor, const DataDescriptor& rangeDescriptor);

        /// default destructor
        ~ZeroOperator() override = default;

    protected:
        /// default copy constructor, hidden from non-derived classes to prevent potential slicing
        ZeroOperator(const ZeroOperator<data_t>&) = default;

        /**
         * @brief apply ZeroOperator A to x, i.e. Ax = 0
         *
         * @param[in] x input DataContainer (in the domain of the operator)
         * @param[out] Ax output DataContainer (in the range of the operator)
         */
        void applyImpl(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const override;

        /**
         * @brief apply the adjoint of the ZeroOperator A to y, i.e. A^ty = 0
         *
         * @param[in] y input DataContainer (in the range of the operator)
         * @param[out] Aty output DataContainer (in the domain of the operator)
         */
        void applyAdjointImpl(const DataContainer<data_t>& y,
                              DataContainer<data_t>& Aty) const override;

        /// implement the polymorphic clone operation
        ZeroOperator<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override;
    };
} // namespace elsa
