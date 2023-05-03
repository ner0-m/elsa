#pragma once

#include "LinearOperator.h"
#include "DetectorDescriptor.h"
#include "VolumeDescriptor.h"
#include "VoxelComputation.h"
#include "Blobs.h"
#include "BSplines.h"

namespace elsa
{
    template <typename data_t = real_t, size_t N = DEFAULT_LUT_SIZE>
    class BlobVoxelProjector : public LinearOperator<data_t>
    {
    public:
        BlobVoxelProjector(const VolumeDescriptor& domainDescriptor,
                           const DetectorDescriptor& rangeDescriptor,
                           data_t radius = blobs::DEFAULT_RADIUS,
                           data_t alpha = blobs::DEFAULT_ALPHA,
                           index_t order = blobs::DEFAULT_ORDER)
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
                throw InvalidArgumentError(
                    "BlobVoxelProjector: rangeDescriptor without any geometry");
            }
        }

    protected:
        void applyImpl(const elsa::DataContainer<data_t>& x,
                       elsa::DataContainer<data_t>& Ax) const override
        {
            if (_dim == 2)
                voxel::forwardVoxel<2>(x, Ax, blob.get_lut(),
                                       voxel::classic_weight_function<1, N, data_t>);
            else
                voxel::forwardVoxel<3>(x, Ax, blob.get_lut(),
                                       voxel::classic_weight_function<2, N, data_t>);
        }

        void applyAdjointImpl(const elsa::DataContainer<data_t>& y,
                              elsa::DataContainer<data_t>& Aty) const override
        {
            if (_dim == 2)
                voxel::backwardVoxel<2>(y, Aty, blob.get_lut(),
                                        voxel::classic_weight_function<1, N, data_t>);
            else
                voxel::backwardVoxel<3>(y, Aty, blob.get_lut(),
                                        voxel::classic_weight_function<2, N, data_t>);
        }

        /// implement the polymorphic comparison operation
        bool isEqual(const elsa::LinearOperator<data_t>& other) const override
        {
            return LinearOperator<data_t>::isEqual(other);
        }

        /// implement the polymorphic comparison operation
        BlobVoxelProjector<data_t, N>* cloneImpl() const override
        {
            return new BlobVoxelProjector(downcast<VolumeDescriptor>(*this->_domainDescriptor),
                                          downcast<DetectorDescriptor>(*this->_rangeDescriptor),
                                          blob.radius(), blob.alpha(), blob.order());
        }

    public:
        const ProjectedBlob<data_t, N> blob;

    private:
        index_t _dim;
    };

    template <typename data_t = real_t, size_t N = DEFAULT_LUT_SIZE>
    class BSplineVoxelProjector : public LinearOperator<data_t>
    {
    public:
        BSplineVoxelProjector(const VolumeDescriptor& domainDescriptor,
                              const DetectorDescriptor& rangeDescriptor,
                              const index_t order = bspline::DEFAULT_ORDER)
            : LinearOperator<data_t>(domainDescriptor, rangeDescriptor),
              _dim(domainDescriptor.getNumberOfDimensions()),
              bspline(_dim, order)
        {
            // sanity checks
            if (_dim < 2 || _dim > 3) {
                throw InvalidArgumentError(
                    "BSplineVoxelProjector: only supporting 2d/3d operations");
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

    protected:
        void applyImpl(const elsa::DataContainer<data_t>& x,
                       elsa::DataContainer<data_t>& Ax) const override
        {
            if (_dim == 2)
                voxel::forwardVoxel<2>(x, Ax, bspline.get_lut(),
                                       voxel::classic_weight_function<1, N, data_t>);
            else
                voxel::forwardVoxel<3>(x, Ax, bspline.get_lut(),
                                       voxel::classic_weight_function<2, N, data_t>);
        }

        void applyAdjointImpl(const elsa::DataContainer<data_t>& y,
                              elsa::DataContainer<data_t>& Aty) const override
        {
            if (_dim == 2)
                voxel::backwardVoxel<2>(y, Aty, bspline.get_lut(),
                                        voxel::classic_weight_function<1, N, data_t>);
            else
                voxel::backwardVoxel<3>(y, Aty, bspline.get_lut(),
                                        voxel::classic_weight_function<2, N, data_t>);
        }

        /// implement the polymorphic comparison operation
        bool isEqual(const elsa::LinearOperator<data_t>& other) const override
        {
            return LinearOperator<data_t>::isEqual(other);
        }

        /// implement the polymorphic comparison operation
        BSplineVoxelProjector<data_t, N>* cloneImpl() const override
        {
            return new BSplineVoxelProjector(downcast<VolumeDescriptor>(*this->_domainDescriptor),
                                             downcast<DetectorDescriptor>(*this->_rangeDescriptor),
                                             bspline.order());
        }

    private:
        index_t _dim;

    public:
        const ProjectedBSpline<data_t, N> bspline;
    };
} // namespace elsa