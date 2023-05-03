#pragma once

#include "LinearOperator.h"
#include "DetectorDescriptor.h"
#include "VolumeDescriptor.h"
#include "VoxelComputation.h"

namespace elsa
{

    template <typename data_t>
    class PhaseContrastProjector : public LinearOperator<data_t>
    {
    public:
        PhaseContrastProjector(const VolumeDescriptor& domainDescriptor,
                               const DetectorDescriptor& rangeDescriptor);

    protected:
        void applyImpl(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const override;

        void applyAdjointImpl(const DataContainer<data_t>& y,
                              DataContainer<data_t>& Aty) const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override;

        /// implement the polymorphic comparison operation
        PhaseContrastProjector<data_t>* cloneImpl() const override;

    private:
        voxel::VoxelLut<real_t, 100> voxelLut;
        index_t _dim;
    };
} // namespace elsa
