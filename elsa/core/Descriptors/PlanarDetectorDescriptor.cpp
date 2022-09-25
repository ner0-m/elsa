#include "PlanarDetectorDescriptor.h"

namespace elsa
{
    PlanarDetectorDescriptor::PlanarDetectorDescriptor(const IndexVector_t& numOfCoeffsPerDim,
                                                       const std::vector<Geometry>& geometryList)
        : DetectorDescriptor(numOfCoeffsPerDim, geometryList)
    {
    }
    PlanarDetectorDescriptor::PlanarDetectorDescriptor(const IndexVector_t& numOfCoeffsPerDim,
                                                       const RealVector_t& spacingPerDim,
                                                       const std::vector<Geometry>& geometryList)
        : DetectorDescriptor(numOfCoeffsPerDim, spacingPerDim, geometryList)
    {
    }

    RealRay_t
        PlanarDetectorDescriptor::computeRayFromDetectorCoord(const RealVector_t& detectorCoord,
                                                              const index_t poseIndex) const
    {
        return DetectorDescriptor::computeRayFromDetectorCoord(detectorCoord, poseIndex);
    }

    bool PlanarDetectorDescriptor::isEqual(const DataDescriptor& other) const
    {
        // PlanarDetectorDescriptor has no data, so just deligate it to base class
        return DetectorDescriptor::isEqual(other);
    }

    PlanarDetectorDescriptor* PlanarDetectorDescriptor::cloneImpl() const
    {
        return new PlanarDetectorDescriptor(getNumberOfCoefficientsPerDimension(),
                                            getSpacingPerDimension(), _geometry);
    }
} // namespace elsa
