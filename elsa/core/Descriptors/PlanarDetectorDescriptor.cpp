#include "PlanarDetectorDescriptor.h"
#include <iostream>

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

    DetectorDescriptor::Ray
        PlanarDetectorDescriptor::computeRayFromDetectorCoord(const RealVector_t& detectorCoord,
                                                              const index_t poseIndex) const
    {
        // Assert that for all dimension of detectorCoord is in bounds and poseIndex can
        // be index in the _geometry. If not the calculation will not be correct, but
        // as this is the hot path, I don't want execptions and unpacking everything
        // We'll just have to ensure, that we don't mess up in our hot path! :-)
        assert((detectorCoord.block(0, 0, getNumberOfDimensions() - 1, 0).array()
                < getNumberOfCoefficientsPerDimension()
                      .block(0, 0, getNumberOfDimensions() - 1, 0)
                      .template cast<real_t>()
                      .array())
                   .all()
               && "PlanarDetectorDescriptor::computeRayToDetector: Assumption detectorCoord in "
                  "bounds, wrong");
        assert(std::make_unsigned_t<std::size_t>(poseIndex) < _geometry.size()
               && "PlanarDetectorDescriptor::computeRayToDetector: Assumption poseIndex smaller "
                  "than number of poses, wrong");

        auto dim = getNumberOfDimensions();

        // get the pose of trajectory
        auto geometry = _geometry[std::make_unsigned_t<index_t>(poseIndex)];

        auto projInvMatrix = geometry.getInverseProjectionMatrix();

        // homogeneous coordinates [p;1], with p in detector space
        RealVector_t homogeneousPixelCoord(dim);
        homogeneousPixelCoord << detectorCoord, 1;

        // Camera center is always the ray origin
        auto ro = geometry.getCameraCenter();

        auto rd = (projInvMatrix * homogeneousPixelCoord) // Matrix-Vector multiplication
                      .head(dim)                          // Transform to non-homogeneous
                      .normalized();                      // normalize vector

        return Ray(ro, rd);
    }

    RealVector_t
        PlanarDetectorDescriptor::computeDetectorCoordFromRay(const Ray& ray,
                                                              const index_t poseIndex) const
    {
        auto dim = getNumberOfDimensions();
        auto geometry = _geometry[static_cast<std::size_t>(poseIndex)];

        auto projMatrix = geometry.getProjectionMatrix();

        // Only take the square matrix part
        auto pixel = (projMatrix.block(0, 0, dim, dim) * ray.direction()).head(dim - 1);

        return pixel;
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
