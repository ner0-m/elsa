#include "PhaseContrastProjector.h"

namespace elsa
{

    template <typename data_t>
    PhaseContrastProjector<data_t>::PhaseContrastProjector(
        const VolumeDescriptor& domainDescriptor, const DetectorDescriptor& rangeDescriptor)
        : LinearOperator<data_t>(domainDescriptor, rangeDescriptor),
          _dim(domainDescriptor.getNumberOfDimensions())
    {
        // sanity checks
        if (_dim < 2 || _dim > 3) {
            throw InvalidArgumentError("VoxelLutProjector: only supporting 2d/3d operations");
        }

        if (_dim != rangeDescriptor.getNumberOfDimensions()) {
            throw InvalidArgumentError(
                "VoxelLutProjector: domain and range dimension need to match");
        }

        if (rangeDescriptor.getNumberOfGeometryPoses() == 0) {
            throw InvalidArgumentError("VoxelLutProjector: rangeDescriptor without any geometry");
        }
    }

    template <typename data_t>
    void PhaseContrastProjector<data_t>::applyImpl(const elsa::DataContainer<data_t>& x,
                                                   elsa::DataContainer<data_t>& Ax) const
    {
        if (_dim == 2)
            voxel::forwardVoxel<2>(x, Ax, voxelLut);
        else
            voxel::forwardVoxel<3>(x, Ax, voxelLut);
    }

    template <typename data_t>
    void PhaseContrastProjector<data_t>::applyAdjointImpl(const elsa::DataContainer<data_t>& y,
                                                          elsa::DataContainer<data_t>& Aty) const
    {
        if (_dim == 2)
            voxel::backwardVoxel<2>(y, Aty, voxelLut);
        else
            voxel::backwardVoxel<3>(y, Aty, voxelLut);
    }

    template <typename data_t>
    bool PhaseContrastProjector<data_t>::isEqual(const elsa::LinearOperator<data_t>& other) const
    {
        return LinearOperator<data_t>::isEqual(other);
    }

    template <typename data_t>
    PhaseContrastProjector<data_t>* PhaseContrastProjector<data_t>::cloneImpl() const
    {
        return new PhaseContrastProjector(downcast<VolumeDescriptor>(*this->_domainDescriptor),
                                          downcast<DetectorDescriptor>(*this->_rangeDescriptor));
    }

    template class PhaseContrastProjector<float>;
    template class PhaseContrastProjector<double>;
}; // namespace elsa
