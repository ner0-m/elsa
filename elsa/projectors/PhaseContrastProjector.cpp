#include "PhaseContrastProjector.h"

namespace elsa
{
    template <typename data_t>
    PhaseContrastBlobVoxelProjector<data_t>::PhaseContrastBlobVoxelProjector(
        const VolumeDescriptor& domainDescriptor, const DetectorDescriptor& rangeDescriptor,
        data_t radius, data_t alpha, index_t order)
        : LinearOperator<data_t>(domainDescriptor, rangeDescriptor),
          blob(radius, alpha, order),
          _dim(domainDescriptor.getNumberOfDimensions())
    {
        // sanity checks
        if (_dim < 2 || _dim > 3) {
            throw InvalidArgumentError(
                "PhaseContrastBlobVoxelProjector: only supporting 2d/3d operations");
        }

        if (_dim != rangeDescriptor.getNumberOfDimensions()) {
            throw InvalidArgumentError(
                "PhaseContrastBlobVoxelProjector: domain and range dimension need to match");
        }

        if (rangeDescriptor.getNumberOfGeometryPoses() == 0) {
            throw InvalidArgumentError(
                "PhaseContrastBlobVoxelProjector: rangeDescriptor without any geometry");
        }
    }

    template <typename data_t>
    void PhaseContrastBlobVoxelProjector<data_t>::applyImpl(const elsa::DataContainer<data_t>& x,
                                                            elsa::DataContainer<data_t>& Ax) const
    {
        if (_dim == 2)
            voxel::forwardVoxel<2>(x, Ax, blob.get_derivative_lut(),
                                   voxel::differential_weight_function_2D<data_t>);
        else
            voxel::forwardVoxel<3>(x, Ax, blob.get_normalized_gradient_lut(),
                                   voxel::differential_weight_function_3D<data_t>);
    }

    template <typename data_t>
    void PhaseContrastBlobVoxelProjector<data_t>::applyAdjointImpl(
        const elsa::DataContainer<data_t>& y, elsa::DataContainer<data_t>& Aty) const
    {
        if (_dim == 2)
            voxel::backwardVoxel<2>(y, Aty, blob.get_derivative_lut(),
                                    voxel::differential_weight_function_2D<data_t>);
        else
            voxel::backwardVoxel<3>(y, Aty, blob.get_normalized_gradient_lut(),
                                    voxel::differential_weight_function_3D<data_t>);
    }

    template <typename data_t>
    PhaseContrastBlobVoxelProjector<data_t>*
        PhaseContrastBlobVoxelProjector<data_t>::cloneImpl() const
    {
        return new PhaseContrastBlobVoxelProjector(
            downcast<VolumeDescriptor>(*this->_domainDescriptor),
            downcast<DetectorDescriptor>(*this->_rangeDescriptor), blob.radius(), blob.alpha(),
            blob.order());
    }

    template <typename data_t>
    bool PhaseContrastBlobVoxelProjector<data_t>::isEqual(
        const elsa::LinearOperator<data_t>& other) const
    {
        return LinearOperator<data_t>::isEqual(other);
    }

    template <typename data_t>
    PhaseContrastBSplineVoxelProjector<data_t>::PhaseContrastBSplineVoxelProjector(
        const VolumeDescriptor& domainDescriptor, const DetectorDescriptor& rangeDescriptor,
        const index_t order)
        : LinearOperator<data_t>(domainDescriptor, rangeDescriptor),
          _dim(domainDescriptor.getNumberOfDimensions()),
          bspline(_dim, order)
    {
        // sanity checks
        if (_dim < 2 || _dim > 3) {
            throw InvalidArgumentError(
                "PhaseContrastBSplineVoxelProjector: only supporting 2d/3d operations");
        }

        if (_dim != rangeDescriptor.getNumberOfDimensions()) {
            throw InvalidArgumentError(
                "PhaseContrastBSplineVoxelProjector: domain and range dimension need to match");
        }

        if (rangeDescriptor.getNumberOfGeometryPoses() == 0) {
            throw InvalidArgumentError(
                "PhaseContrastBSplineVoxelProjector: rangeDescriptor without any geometry");
        }
    }

    template <typename data_t>
    void
        PhaseContrastBSplineVoxelProjector<data_t>::applyImpl(const elsa::DataContainer<data_t>& x,
                                                              elsa::DataContainer<data_t>& Ax) const
    {
        if (_dim == 2)
            voxel::forwardVoxel<2>(x, Ax, bspline.get_derivative_lut(),
                                   voxel::differential_weight_function_2D<data_t>);
        else
            voxel::forwardVoxel<3>(x, Ax, bspline.get_normalized_gradient_lut(),
                                   voxel::differential_weight_function_3D<data_t>);
    }

    template <typename data_t>
    void PhaseContrastBSplineVoxelProjector<data_t>::applyAdjointImpl(
        const elsa::DataContainer<data_t>& y, elsa::DataContainer<data_t>& Aty) const
    {
        if (_dim == 2)
            voxel::backwardVoxel<2>(y, Aty, bspline.get_derivative_lut(),
                                    voxel::differential_weight_function_2D<data_t>);
        else
            voxel::backwardVoxel<3>(y, Aty, bspline.get_normalized_gradient_lut(),
                                    voxel::differential_weight_function_3D<data_t>);
    }

    template <typename data_t>
    PhaseContrastBSplineVoxelProjector<data_t>*
        PhaseContrastBSplineVoxelProjector<data_t>::cloneImpl() const
    {
        return new PhaseContrastBSplineVoxelProjector(
            downcast<VolumeDescriptor>(*this->_domainDescriptor),
            downcast<DetectorDescriptor>(*this->_rangeDescriptor), bspline.order());
    }

    template <typename data_t>
    bool PhaseContrastBSplineVoxelProjector<data_t>::isEqual(
        const elsa::LinearOperator<data_t>& other) const
    {
        return LinearOperator<data_t>::isEqual(other);
    }

    template class PhaseContrastBSplineVoxelProjector<float>;
    template class PhaseContrastBSplineVoxelProjector<double>;
    template class PhaseContrastBlobVoxelProjector<float>;
    template class PhaseContrastBlobVoxelProjector<double>;
}; // namespace elsa
