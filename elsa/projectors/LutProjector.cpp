#include "LutProjector.h"
#include "Timer.h"
#include "SliceTraversal.h"
#include "Assertions.h"

namespace elsa
{
    template <typename data_t>
    BlobProjector<data_t>::BlobProjector(const VolumeDescriptor& domainDescriptor,
                                         const DetectorDescriptor& rangeDescriptor, data_t radius,
                                         data_t alpha, index_t order)
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
        : BlobProjector(domainDescriptor, rangeDescriptor, 2.f, 10.83f, 2.f)
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
    BSplineProjector<data_t>::BSplineProjector(const VolumeDescriptor& domainDescriptor,
                                               const DetectorDescriptor& rangeDescriptor,
                                               index_t order)
        : LutProjector<data_t, BSplineProjector<data_t>>(domainDescriptor, rangeDescriptor),
          lut_(domainDescriptor.getNumberOfDimensions(), order)
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
        : BSplineProjector(domainDescriptor, rangeDescriptor, 2)
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

    template class BSplineProjector<float>;
    template class BSplineProjector<double>;
} // namespace elsa
