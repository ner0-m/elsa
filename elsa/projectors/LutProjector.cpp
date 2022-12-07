#include "LutProjector.h"
#include "Timer.h"
#include "SliceTraversal.h"
#include "Assertions.h"

Eigen::IOFormat vecFmt(10, 0, ", ", ", ", "", "", "[", "]");
Eigen::IOFormat matFmt(10, 0, ", ", "\n", "\t\t[", "]");

namespace elsa
{
    template <typename data_t>
    BlobProjector<data_t>::BlobProjector(data_t radius, data_t alpha, data_t order,
                                         const VolumeDescriptor& domainDescriptor,
                                         const DetectorDescriptor& rangeDescriptor)
        : LutProjector<data_t, BlobProjector<data_t>>(domainDescriptor, rangeDescriptor),
          lut_(radius, alpha, order)
    {
        // sanity checks
        auto dim = domainDescriptor.getNumberOfDimensions();
        if (dim < 2 || dim > 3) {
            throw InvalidArgumentError("BlobProjector: only supporting 2d/3d operations");
        }

        if (dim != rangeDescriptor.getNumberOfDimensions()) {
            throw InvalidArgumentError("BlobProjector: domain and range dimension need to match");
        }

        if (rangeDescriptor.getNumberOfGeometryPoses() == 0) {
            throw InvalidArgumentError("BlobProjector: rangeDescriptor without any geometry");
        }
    }

    template <typename data_t>
    BlobProjector<data_t>::BlobProjector(const VolumeDescriptor& domainDescriptor,
                                         const DetectorDescriptor& rangeDescriptor)
        : BlobProjector(2, 10.83, 2, domainDescriptor, rangeDescriptor)
    {
    }

    template <typename data_t>
    BlobProjector<data_t>* BlobProjector<data_t>::_cloneImpl() const
    {
        return new BlobProjector(downcast<VolumeDescriptor>(*this->_domainDescriptor),
                                 downcast<DetectorDescriptor>(*this->_rangeDescriptor));
    }

    template <typename data_t>
    bool BlobProjector<data_t>::_isEqual(const LinearOperator<data_t>& other) const
    {
        if (!Base::isEqual(other))
            return false;

        auto otherOp = downcast_safe<BlobProjector>(&other);
        return static_cast<bool>(otherOp);
    }

    template <typename data_t>
    VoxelBlobProjector<data_t>::VoxelBlobProjector(data_t radius, data_t alpha, data_t order,
                                                   const VolumeDescriptor& domainDescriptor,
                                                   const DetectorDescriptor& rangeDescriptor)
        : LutProjector<data_t, VoxelBlobProjector<data_t>>(domainDescriptor, rangeDescriptor),
          lut_(radius, alpha, order)
    {
        // sanity checks
        auto dim = domainDescriptor.getNumberOfDimensions();
        if (dim < 2 || dim > 3) {
            throw InvalidArgumentError("BlobProjector: only supporting 2d/3d operations");
        }

        if (dim != rangeDescriptor.getNumberOfDimensions()) {
            throw InvalidArgumentError("BlobProjector: domain and range dimension need to match");
        }

        if (rangeDescriptor.getNumberOfGeometryPoses() == 0) {
            throw InvalidArgumentError("BlobProjector: rangeDescriptor without any geometry");
        }
    }

    template <typename data_t>
    VoxelBlobProjector<data_t>::VoxelBlobProjector(const VolumeDescriptor& domainDescriptor,
                                                   const DetectorDescriptor& rangeDescriptor)
        : VoxelBlobProjector(2, 10.83, 2, domainDescriptor, rangeDescriptor)
    {
    }

    template <typename data_t>
    VoxelBlobProjector<data_t>* VoxelBlobProjector<data_t>::_cloneImpl() const
    {
        return new VoxelBlobProjector(downcast<VolumeDescriptor>(*this->_domainDescriptor),
                                      downcast<DetectorDescriptor>(*this->_rangeDescriptor));
    }

    template <typename data_t>
    bool VoxelBlobProjector<data_t>::_isEqual(const LinearOperator<data_t>& other) const
    {
        if (!Base::isEqual(other))
            return false;

        auto otherOp = downcast_safe<VoxelBlobProjector>(&other);
        return static_cast<bool>(otherOp);
    }
    template <typename data_t>
    void VoxelBlobProjector<data_t>::forward(const BoundingBox aabb, const DataContainer<data_t>& x,
                                             DataContainer<data_t>& Ax) const
    {
        // Ax is the detector
        // x is the volume
        // given x, compute Ax

        const DetectorDescriptor& detectorDesc =
            downcast<DetectorDescriptor>(Ax.getDataDescriptor());
        const auto numGeoms = detectorDesc.getNumberOfGeometryPoses();

        // loop over geometries
#pragma omp parallel for
        for (index_t geomIndex = 0; geomIndex < numGeoms; geomIndex++) {
            // loop over voxels
            for (index_t domainIndex = 0; domainIndex < x.getSize(); ++domainIndex) {
                auto coord = x.getDataDescriptor().getCoordinateFromIndex(domainIndex);
                auto dim = x.getDataDescriptor().getNumberOfDimensions();
                // Cast to real_t and shift to center of pixel
                RealVector_t volumeCoord = coord.template cast<real_t>().array() + 0.5;

                auto voxelWeight = x[domainIndex];

                auto [detectorCoord, scaling] =
                    detectorDesc.projectAndScaleVoxelOnDetector(volumeCoord, geomIndex);

                // find all detector pixels that are hit
                auto radiusOnDetector = static_cast<index_t>(std::ceil(lut_.radius() * scaling));
                radiusOnDetector = 0;

                // TODO hardcoded for 2D and offset not completely correct
                // assumes that detectorCoord is already at the center of a detector pixel
                // find detector pixels for this center
                for (index_t i = -radiusOnDetector; i <= radiusOnDetector * 2; i++) {
                    RealVector_t offset(2);
                    offset << i, 0;
                    index_t detector_index = detectorDesc.getIndexFromCoordinate(
                        (detectorCoord + offset).template cast<index_t>());
                    if (detector_index >= 0 && detector_index < Ax.getSize()) {
                        Ax[detector_index] += weight(offset.norm()) * voxelWeight;
                    }
                }
            }
        }
    }

    template <typename data_t>
    void VoxelBlobProjector<data_t>::backward(const BoundingBox aabb,
                                              const DataContainer<data_t>& y,
                                              DataContainer<data_t>& Aty) const
    {
        // y is the detector
        // Aty is the volume
        // given y, compute Aty

        const DetectorDescriptor& detectorDesc =
            downcast<DetectorDescriptor>(y.getDataDescriptor());
        const auto numGeoms = detectorDesc.getNumberOfGeometryPoses();

        // loop over geometries
        for (index_t geomIndex = 0; geomIndex < numGeoms; geomIndex++) {
            // loop over voxels
            for (index_t domainIndex = 0; domainIndex < Aty.getSize(); ++domainIndex) {
                auto coord = Aty.getDataDescriptor().getCoordinateFromIndex(domainIndex);
                auto dim = Aty.getDataDescriptor().getNumberOfDimensions();
                // Cast to real_t and shift to center of pixel
                RealVector_t volumeCoord = coord.template cast<real_t>().array() + 0.5;

                auto [detectorCoord, scaling] =
                    detectorDesc.projectAndScaleVoxelOnDetector(volumeCoord, geomIndex);

                // find all detector pixels that are hit
                auto radiusOnDetector = static_cast<index_t>(std::ceil(lut_.radius() * scaling));
                radiusOnDetector = 0;

                // find detector pixels for this center
                for (index_t i = -radiusOnDetector; i <= radiusOnDetector * 2; i++) {
                    RealVector_t offset(2);
                    offset << i, 0;
                    index_t detector_index = detectorDesc.getIndexFromCoordinate(
                        (detectorCoord + offset).template cast<index_t>());
                    if (detector_index >= 0 && detector_index < y.getSize()) {
                        Aty[domainIndex] += weight(offset.norm()) * y[detector_index];
                    }
                }
            }
        }
    }

    template <typename data_t>
    DifferentialBlobProjector<data_t>::DifferentialBlobProjector(
        data_t radius, data_t alpha, data_t order, const VolumeDescriptor& domainDescriptor,
        const DetectorDescriptor& rangeDescriptor)
        : LutProjector<data_t, DifferentialBlobProjector<data_t>>(domainDescriptor,
                                                                  rangeDescriptor),
          lut_(radius, alpha, order)
    {
        // sanity checks
        auto dim = domainDescriptor.getNumberOfDimensions();
        if (dim < 2 || dim > 3) {
            throw InvalidArgumentError(
                "DifferentialBlobProjector: only supporting 2d/3d operations");
        }

        if (dim != rangeDescriptor.getNumberOfDimensions()) {
            throw InvalidArgumentError(
                "DifferentialBlobProjector: domain and range dimension need to match");
        }

        if (rangeDescriptor.getNumberOfGeometryPoses() == 0) {
            throw InvalidArgumentError(
                "DifferentialBlobProjector: rangeDescriptor without any geometry");
        }
    }

    template <typename data_t>
    DifferentialBlobProjector<data_t>::DifferentialBlobProjector(
        const VolumeDescriptor& domainDescriptor, const DetectorDescriptor& rangeDescriptor)
        : DifferentialBlobProjector(2, 10.83, 2, domainDescriptor, rangeDescriptor)
    {
    }

    template <typename data_t>
    DifferentialBlobProjector<data_t>* DifferentialBlobProjector<data_t>::_cloneImpl() const
    {
        return new DifferentialBlobProjector(downcast<VolumeDescriptor>(*this->_domainDescriptor),
                                             downcast<DetectorDescriptor>(*this->_rangeDescriptor));
    }

    template <typename data_t>
    bool DifferentialBlobProjector<data_t>::_isEqual(const LinearOperator<data_t>& other) const
    {
        if (!Base::isEqual(other))
            return false;

        auto otherOp = downcast_safe<DifferentialBlobProjector>(&other);
        return static_cast<bool>(otherOp);
    }

    template <typename data_t>
    BSplineProjector<data_t>::BSplineProjector(data_t degree,
                                               const VolumeDescriptor& domainDescriptor,
                                               const DetectorDescriptor& rangeDescriptor)
        : LutProjector<data_t, BSplineProjector<data_t>>(domainDescriptor, rangeDescriptor),
          lut_(domainDescriptor.getNumberOfDimensions(), degree)
    {
        // sanity checks
        auto dim = domainDescriptor.getNumberOfDimensions();
        if (dim < 2 || dim > 3) {
            throw InvalidArgumentError("BSplineProjector: only supporting 2d/3d operations");
        }

        if (dim != rangeDescriptor.getNumberOfDimensions()) {
            throw InvalidArgumentError(
                "BSplineProjector: domain and range dimension need to match");
        }

        if (rangeDescriptor.getNumberOfGeometryPoses() == 0) {
            throw InvalidArgumentError("BSplineProjector: rangeDescriptor without any geometry");
        }
    }

    template <typename data_t>
    BSplineProjector<data_t>::BSplineProjector(const VolumeDescriptor& domainDescriptor,
                                               const DetectorDescriptor& rangeDescriptor)
        : BSplineProjector(2, domainDescriptor, rangeDescriptor)
    {
    }

    template <typename data_t>
    data_t BSplineProjector<data_t>::weight(data_t distance) const
    {
        return lut_(distance);
    }

    template <typename data_t>
    index_t BSplineProjector<data_t>::support() const
    {
        return static_cast<index_t>(2);
    }

    template <typename data_t>
    BSplineProjector<data_t>* BSplineProjector<data_t>::_cloneImpl() const
    {
        return new BSplineProjector(downcast<VolumeDescriptor>(*this->_domainDescriptor),
                                    downcast<DetectorDescriptor>(*this->_rangeDescriptor));
    }

    template <typename data_t>
    bool BSplineProjector<data_t>::_isEqual(const LinearOperator<data_t>& other) const
    {
        if (!Base::isEqual(other))
            return false;

        auto otherOp = downcast_safe<BSplineProjector>(&other);
        return static_cast<bool>(otherOp);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class BlobProjector<float>;
    template class BlobProjector<double>;

    template class VoxelBlobProjector<float>;
    template class VoxelBlobProjector<double>;

    template class DifferentialBlobProjector<float>;
    template class DifferentialBlobProjector<double>;

    template class BSplineProjector<float>;
    template class BSplineProjector<double>;
} // namespace elsa
