#include "VoxelProjector.h"
#include "Timer.h"
#include "Assertions.h"

namespace elsa
{

    template <typename data_t>
    BlobVoxelProjector<data_t>::BlobVoxelProjector(const VolumeDescriptor& domainDescriptor,
                                                   const DetectorDescriptor& rangeDescriptor,
                                                   data_t radius, data_t alpha, index_t order)
        : VoxelProjector<data_t, BlobVoxelProjector<data_t>>(domainDescriptor, rangeDescriptor),
          lut_(radius, alpha, order)
    {
        // sanity checks
        auto dim = domainDescriptor.getNumberOfDimensions();
        if (dim < 2 || dim > 3) {
            throw InvalidArgumentError("BlobVoxelProjector: only supporting 2d/3d operations");
        }

        if (dim != rangeDescriptor.getNumberOfDimensions()) {
            throw InvalidArgumentError(
                "BlobVoxelProjector: domain and range dimension need to match");
        }

        if (rangeDescriptor.getNumberOfGeometryPoses() == 0) {
            throw InvalidArgumentError("BlobVoxelProjector: rangeDescriptor without any geometry");
        }
    }

    template <typename data_t>
    BlobVoxelProjector<data_t>::BlobVoxelProjector(const VolumeDescriptor& domainDescriptor,
                                                   const DetectorDescriptor& rangeDescriptor)
        : BlobVoxelProjector(domainDescriptor, rangeDescriptor, 2, 10.83, 2)
    {
    }

    template <typename data_t>
    BlobVoxelProjector<data_t>* BlobVoxelProjector<data_t>::_cloneImpl() const
    {
        return new BlobVoxelProjector(downcast<VolumeDescriptor>(*this->_domainDescriptor),
                                      downcast<DetectorDescriptor>(*this->_rangeDescriptor),
                                      this->lut_.radius(), this->lut_.alpha(), this->lut_.order());
    }

    template <typename data_t>
    bool BlobVoxelProjector<data_t>::_isEqual(const LinearOperator<data_t>& other) const
    {
        if (!Base::isEqual(other))
            return false;

        auto otherOp = downcast_safe<BlobVoxelProjector>(&other);
        return static_cast<bool>(otherOp);
    }

    template <typename data_t>
    PhaseContrastBlobVoxelProjector<data_t>::PhaseContrastBlobVoxelProjector(
        const VolumeDescriptor& domainDescriptor, const DetectorDescriptor& rangeDescriptor,
        data_t radius, data_t alpha, index_t order)
        : VoxelProjector<data_t, PhaseContrastBlobVoxelProjector<data_t>>(domainDescriptor,
                                                                          rangeDescriptor),
          lut_(radius, alpha, order),
          lut3D_(radius, alpha, order)
    {
        // sanity checks
        auto dim = domainDescriptor.getNumberOfDimensions();
        if (dim < 2 || dim > 3) {
            throw InvalidArgumentError(
                "PhaseContrastBlobVoxelProjector: only supporting 2d/3d operations");
        }

        if (dim != rangeDescriptor.getNumberOfDimensions()) {
            throw InvalidArgumentError(
                "PhaseContrastBlobVoxelProjector: domain and range dimension need to match");
        }

        if (rangeDescriptor.getNumberOfGeometryPoses() == 0) {
            throw InvalidArgumentError(
                "PhaseContrastBlobVoxelProjector: rangeDescriptor without any geometry");
        }
    }

    template <typename data_t>
    PhaseContrastBlobVoxelProjector<data_t>::PhaseContrastBlobVoxelProjector(
        const VolumeDescriptor& domainDescriptor, const DetectorDescriptor& rangeDescriptor)
        : PhaseContrastBlobVoxelProjector(domainDescriptor, rangeDescriptor, 2, 10.83, 2)
    {
    }

    template <typename data_t>
    PhaseContrastBlobVoxelProjector<data_t>*
        PhaseContrastBlobVoxelProjector<data_t>::_cloneImpl() const
    {
        return new PhaseContrastBlobVoxelProjector(
            downcast<VolumeDescriptor>(*this->_domainDescriptor),
            downcast<DetectorDescriptor>(*this->_rangeDescriptor), this->lut_.radius(),
            this->lut_.alpha(), this->lut_.order());
    }

    template <typename data_t>
    bool
        PhaseContrastBlobVoxelProjector<data_t>::_isEqual(const LinearOperator<data_t>& other) const
    {
        if (!Base::isEqual(other))
            return false;

        auto otherOp = downcast_safe<PhaseContrastBlobVoxelProjector>(&other);
        return static_cast<bool>(otherOp);
    }

    template <typename data_t>
    BSplineVoxelProjector<data_t>::BSplineVoxelProjector(const VolumeDescriptor& domainDescriptor,
                                                         const DetectorDescriptor& rangeDescriptor,
                                                         index_t order)
        : VoxelProjector<data_t, BSplineVoxelProjector<data_t>>(domainDescriptor, rangeDescriptor),
          lut_(domainDescriptor.getNumberOfDimensions(), order)
    {
        // sanity checks
        auto dim = domainDescriptor.getNumberOfDimensions();
        if (dim < 2 || dim > 3) {
            throw InvalidArgumentError(
                "PhaseContrastBlobVoxelProjector: only supporting 2d/3d operations");
        }

        if (dim != rangeDescriptor.getNumberOfDimensions()) {
            throw InvalidArgumentError(
                "PhaseContrastBlobVoxelProjector: domain and range dimension need to match");
        }

        if (rangeDescriptor.getNumberOfGeometryPoses() == 0) {
            throw InvalidArgumentError(
                "PhaseContrastBlobVoxelProjector: rangeDescriptor without any geometry");
        }
    }

    template <typename data_t>
    BSplineVoxelProjector<data_t>::BSplineVoxelProjector(const VolumeDescriptor& domainDescriptor,
                                                         const DetectorDescriptor& rangeDescriptor)
        : BSplineVoxelProjector(domainDescriptor, rangeDescriptor, 3)
    {
    }

    template <typename data_t>
    BSplineVoxelProjector<data_t>* BSplineVoxelProjector<data_t>::_cloneImpl() const
    {
        return new BSplineVoxelProjector(downcast<VolumeDescriptor>(*this->_domainDescriptor),
                                         downcast<DetectorDescriptor>(*this->_rangeDescriptor),
                                         lut_.order());
    }

    template <typename data_t>
    bool BSplineVoxelProjector<data_t>::_isEqual(const LinearOperator<data_t>& other) const
    {
        if (!Base::isEqual(other))
            return false;

        auto otherOp = downcast_safe<BSplineVoxelProjector>(&other);
        return static_cast<bool>(otherOp);
    }

    template <typename data_t>
    PhaseContrastBSplineVoxelProjector<data_t>::PhaseContrastBSplineVoxelProjector(
        const VolumeDescriptor& domainDescriptor, const DetectorDescriptor& rangeDescriptor,
        index_t order)
        : VoxelProjector<data_t, PhaseContrastBSplineVoxelProjector<data_t>>(domainDescriptor,
                                                                             rangeDescriptor),
          lut_(domainDescriptor.getNumberOfDimensions(), order),
          lut3D_(domainDescriptor.getNumberOfDimensions(), order)
    {
        // sanity checks
        auto dim = domainDescriptor.getNumberOfDimensions();
        if (dim < 2 || dim > 3) {
            throw InvalidArgumentError(
                "PhaseContrastBlobVoxelProjector: only supporting 2d/3d operations");
        }

        if (dim != rangeDescriptor.getNumberOfDimensions()) {
            throw InvalidArgumentError(
                "PhaseContrastBlobVoxelProjector: domain and range dimension need to match");
        }

        if (rangeDescriptor.getNumberOfGeometryPoses() == 0) {
            throw InvalidArgumentError(
                "PhaseContrastBlobVoxelProjector: rangeDescriptor without any geometry");
        }
    }

    template <typename data_t>
    PhaseContrastBSplineVoxelProjector<data_t>::PhaseContrastBSplineVoxelProjector(
        const VolumeDescriptor& domainDescriptor, const DetectorDescriptor& rangeDescriptor)
        : PhaseContrastBSplineVoxelProjector(domainDescriptor, rangeDescriptor, 3)
    {
    }

    template <typename data_t>
    PhaseContrastBSplineVoxelProjector<data_t>*
        PhaseContrastBSplineVoxelProjector<data_t>::_cloneImpl() const
    {
        return new PhaseContrastBSplineVoxelProjector(
            downcast<VolumeDescriptor>(*this->_domainDescriptor),
            downcast<DetectorDescriptor>(*this->_rangeDescriptor), lut_.order());
    }

    template <typename data_t>
    bool PhaseContrastBSplineVoxelProjector<data_t>::_isEqual(
        const LinearOperator<data_t>& other) const
    {
        if (!Base::isEqual(other))
            return false;

        auto otherOp = downcast_safe<PhaseContrastBSplineVoxelProjector>(&other);
        return static_cast<bool>(otherOp);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class BlobVoxelProjector<float>;
    template class BlobVoxelProjector<double>;

    template class PhaseContrastBlobVoxelProjector<float>;
    template class PhaseContrastBlobVoxelProjector<double>;

    template class BSplineVoxelProjector<float>;
    template class BSplineVoxelProjector<double>;

    template class PhaseContrastBSplineVoxelProjector<float>;
    template class PhaseContrastBSplineVoxelProjector<double>;
} // namespace elsa