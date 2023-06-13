#include "VoxelProjector.h"
#include "Timer.h"
#include "Assertions.h"

namespace elsa
{
    template <typename data_t>
    BlobVoxelProjector<data_t>::BlobVoxelProjector(const VolumeDescriptor& domainDescriptor,
                                                   const DetectorDescriptor& rangeDescriptor,
                                                   data_t radius, data_t alpha, index_t order)
        : LinearOperator<data_t>(domainDescriptor, rangeDescriptor),
          blob(radius, alpha, order),
          _dim(domainDescriptor.getNumberOfDimensions())
    {
        // sanity checks
        if (_dim < 2 || _dim > 3) {
            throw InvalidArgumentError("BlobVoxelProjector: only supporting 2d/3d operations");
        }

        if (_dim != rangeDescriptor.getNumberOfDimensions()) {
            throw InvalidArgumentError(
                "BlobVoxelProjector: domain and range dimension need to match");
        }

        if (rangeDescriptor.getNumberOfGeometryPoses() == 0) {
            throw InvalidArgumentError("BlobVoxelProjector: rangeDescriptor without any geometry");
        }
    }

    template <typename data_t>
    void BlobVoxelProjector<data_t>::applyImpl(const elsa::DataContainer<data_t>& x,
                                               elsa::DataContainer<data_t>& Ax) const
    {
        if (_dim == 2)
            voxel::forwardVoxel<2>(x, Ax, blob.get_lut(),
                                   voxel::classic_weight_function<1, data_t>);
        else
            voxel::forwardVoxel<3>(x, Ax, blob.get_lut(),
                                   voxel::classic_weight_function<2, data_t>);
    }

    template <typename data_t>
    void BlobVoxelProjector<data_t>::applyAdjointImpl(const elsa::DataContainer<data_t>& y,
                                                      elsa::DataContainer<data_t>& Aty) const
    {
        if (_dim == 2)
            voxel::backwardVoxel<2>(y, Aty, blob.get_lut(),
                                    voxel::classic_weight_function<1, data_t>);
        else
            voxel::backwardVoxel<3>(y, Aty, blob.get_lut(),
                                    voxel::classic_weight_function<2, data_t>);
    }

    template <typename data_t>
    bool BlobVoxelProjector<data_t>::isEqual(const elsa::LinearOperator<data_t>& other) const
    {
        return LinearOperator<data_t>::isEqual(other);
    }

    template <typename data_t>
    BlobVoxelProjector<data_t>* BlobVoxelProjector<data_t>::cloneImpl() const
    {
        return new BlobVoxelProjector(downcast<VolumeDescriptor>(*this->_domainDescriptor),
                                      downcast<DetectorDescriptor>(*this->_rangeDescriptor),
                                      blob.radius(), blob.alpha(), blob.order());
    }

    template <typename data_t>
    BSplineVoxelProjector<data_t>::BSplineVoxelProjector(const VolumeDescriptor& domainDescriptor,
                                                         const DetectorDescriptor& rangeDescriptor,
                                                         const index_t order)
        : LinearOperator<data_t>(domainDescriptor, rangeDescriptor),
          _dim(domainDescriptor.getNumberOfDimensions()),
          bspline(_dim, order)
    {
        // sanity checks
        if (_dim < 2 || _dim > 3) {
            throw InvalidArgumentError("BSplineVoxelProjector: only supporting 2d/3d operations");
        }

        if (_dim != rangeDescriptor.getNumberOfDimensions()) {
            throw InvalidArgumentError(
                "BSplineVoxelProjector: domain and range dimension need to match");
        }

        if (rangeDescriptor.getNumberOfGeometryPoses() == 0) {
            throw InvalidArgumentError(
                "BSplineVoxelProjector: rangeDescriptor without any geometry");
        }
    }

    template <typename data_t>
    void BSplineVoxelProjector<data_t>::applyImpl(const elsa::DataContainer<data_t>& x,
                                                  elsa::DataContainer<data_t>& Ax) const
    {
        if (_dim == 2)
            voxel::forwardVoxel<2>(x, Ax, bspline.get_lut(),
                                   voxel::classic_weight_function<1, data_t>);
        else
            voxel::forwardVoxel<3>(x, Ax, bspline.get_lut(),
                                   voxel::classic_weight_function<2, data_t>);
    }

    template <typename data_t>
    void BSplineVoxelProjector<data_t>::applyAdjointImpl(const elsa::DataContainer<data_t>& y,
                                                         elsa::DataContainer<data_t>& Aty) const
    {
        if (_dim == 2)
            voxel::backwardVoxel<2>(y, Aty, bspline.get_lut(),
                                    voxel::classic_weight_function<1, data_t>);
        else
            voxel::backwardVoxel<3>(y, Aty, bspline.get_lut(),
                                    voxel::classic_weight_function<2, data_t>);
    }

    template <typename data_t>
    bool BSplineVoxelProjector<data_t>::isEqual(const elsa::LinearOperator<data_t>& other) const
    {
        return LinearOperator<data_t>::isEqual(other);
    }

    template <typename data_t>
    BSplineVoxelProjector<data_t>* BSplineVoxelProjector<data_t>::cloneImpl() const
    {
        return new BSplineVoxelProjector(downcast<VolumeDescriptor>(*this->_domainDescriptor),
                                         downcast<DetectorDescriptor>(*this->_rangeDescriptor),
                                         bspline.order());
    }

    template class BlobVoxelProjector<float>;
    template class BlobVoxelProjector<double>;
    template class BSplineVoxelProjector<float>;
    template class BSplineVoxelProjector<double>;
}; // namespace elsa