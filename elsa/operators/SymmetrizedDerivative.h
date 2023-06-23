#pragma once

#include "LinearOperator.h"

#include <vector>
#include <iostream>

namespace elsa
{

    template <typename data_t = real_t>
    class SymmetrizedDerivative : public LinearOperator<data_t>
    {
    public:
        SymmetrizedDerivative(const DataDescriptor& domainDescriptor);

        ~SymmetrizedDerivative() override = default;

    protected:
        /// default copy constructor, hidden from non-derived classes to prevent potential slicing
        SymmetrizedDerivative(const SymmetrizedDerivative<data_t>&) = default;

        /// apply the finite differences operator
        void applyImpl(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const override;

        /// apply the adjoint of the finite differences operator
        void applyAdjointImpl(const DataContainer<data_t>& y,
                              DataContainer<data_t>& Aty) const override;

        /// implement the polymorphic clone operation
        SymmetrizedDerivative<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override;

    private:

        std::unique_ptr<LinearOperator<data_t>> core_;
        std::unique_ptr<LinearOperator<data_t>> scaling_;
/*
        /// precompute some helper variables to optimize speed
        void precomputeHelpers();

        IndexVector_t _coordDiff;  /// precomputed helper for coordinate diffs
        IndexVector_t _coordDelta; /// precomputed helper for coordinate deltas
*/
    };
} // namespace elsa
