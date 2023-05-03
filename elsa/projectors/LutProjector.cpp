#include "LutProjector.h"
#include "Timer.h"
#include "SliceTraversal.h"
#include "Assertions.h"

namespace elsa
{
    template <typename data_t, size_t N>
    BlobProjector<data_t, N>::BlobProjector(const VolumeDescriptor& domainDescriptor,
                                            const DetectorDescriptor& rangeDescriptor,
                                            data_t radius, data_t alpha, index_t order)
        : LutProjector<data_t, BlobProjector<data_t>>(domainDescriptor, rangeDescriptor),
          blob_(radius, alpha, order)
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

    template <typename data_t, size_t N>
    BlobProjector<data_t, N>* BlobProjector<data_t, N>::_cloneImpl() const
    {
        return new BlobProjector(downcast<VolumeDescriptor>(*this->_domainDescriptor),
                                 downcast<DetectorDescriptor>(*this->_rangeDescriptor));
    }

    template <typename data_t, size_t N>
    bool BlobProjector<data_t, N>::_isEqual(const LinearOperator<data_t>& other) const
    {
        if (!Base::isEqual(other))
            return false;

        auto otherOp = downcast_safe<BlobProjector>(&other);
        return static_cast<bool>(otherOp);
    }

    template <typename data_t, size_t N>
    BSplineProjector<data_t, N>::BSplineProjector(const VolumeDescriptor& domainDescriptor,
                                                  const DetectorDescriptor& rangeDescriptor,
                                                  index_t order)
        : LutProjector<data_t, BSplineProjector<data_t>>(domainDescriptor, rangeDescriptor),
          bspline_(domainDescriptor.getNumberOfDimensions(), order)
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

    template <typename data_t, size_t N>
    data_t BSplineProjector<data_t, N>::weight(data_t distance) const
    {
        return bspline_.get_lut()(distance);
    }

    template <typename data_t, size_t N>
    index_t BSplineProjector<data_t, N>::support() const
    {
        return static_cast<index_t>(2);
    }

    template <typename data_t, size_t N>
    BSplineProjector<data_t, N>* BSplineProjector<data_t, N>::_cloneImpl() const
    {
        return new BSplineProjector(downcast<VolumeDescriptor>(*this->_domainDescriptor),
                                    downcast<DetectorDescriptor>(*this->_rangeDescriptor));
    }

    template <typename data_t, size_t N>
    bool BSplineProjector<data_t, N>::_isEqual(const LinearOperator<data_t>& other) const
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
