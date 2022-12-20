#include "XGIDetectorDescriptor.h"
#include "TypeCasts.hpp"

namespace elsa
{
    XGIDetectorDescriptor::XGIDetectorDescriptor(const IndexVector_t& numOfCoeffsPerDim,
                                                 const RealVector_t& spacingPerDim,
                                                 const std::vector<Geometry>& geometryList,
                                                 const XGIDetectorDescriptor::DirVec& sensDir,
                                                 bool isParallelBeam)
        : DetectorDescriptor(numOfCoeffsPerDim, spacingPerDim, geometryList),
          _sensDir(sensDir.normalized()),
          _isParallelBeam(isParallelBeam)
    {
    }

    const XGIDetectorDescriptor::DirVec& XGIDetectorDescriptor::getSensDir() const
    {
        return _sensDir;
    }

    bool XGIDetectorDescriptor::isParallelBeam() const
    {
        return _isParallelBeam;
    }

    RealRay_t XGIDetectorDescriptor::computeRayFromDetectorCoord(const RealVector_t& detectorCoord,
                                                                 const index_t poseIndex) const
    {
        return DetectorDescriptor::computeRayFromDetectorCoord(detectorCoord, poseIndex);
    }

    bool XGIDetectorDescriptor::isEqual(const DataDescriptor& other) const
    {
        const XGIDetectorDescriptor* otherPtr{
            downcast_safe<const XGIDetectorDescriptor, const DataDescriptor>(&other)};

        return (otherPtr != nullptr && _sensDir == otherPtr->_sensDir
                && _isParallelBeam == otherPtr->_isParallelBeam
                && DetectorDescriptor::isEqual(other));
    }

    XGIDetectorDescriptor* XGIDetectorDescriptor::cloneImpl() const
    {
        return new XGIDetectorDescriptor(getNumberOfCoefficientsPerDimension(),
                                         getSpacingPerDimension(), getGeometry(), getSensDir(),
                                         isParallelBeam());
    }
} // namespace elsa
