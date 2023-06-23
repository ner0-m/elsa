#pragma once

#include "LinearOperator.h"
#include <iostream>

namespace elsa
{

    template <typename data_t = real_t>
    class EmptyTransform : public LinearOperator<data_t>
    {
    public:

        explicit EmptyTransform(const DataDescriptor& domainDescriptor, const DataDescriptor& rangeDescriptor);

        ~EmptyTransform() override = default;

    protected:
        /// default copy constructor, hidden from non-derived classes to prevent potential slicing
        EmptyTransform(const EmptyTransform<data_t>&) = default;

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
        EmptyTransform<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override;
    };
} // namespace elsa
