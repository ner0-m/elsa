#include "CurvedDetectorDescriptor.h"
#include <iostream>
#include <type_traits>

#include "Logger.h"

namespace elsa
{
    CurvedDetectorDescriptor::CurvedDetectorDescriptor(const IndexVector_t& numOfCoeffsPerDim,
                                                       const std::vector<Geometry>& geometryList,
                                                       real_t scale)
        : DetectorDescriptor(numOfCoeffsPerDim, geometryList), _scale(scale)
    {
        // calculate length to , using Law of Cosine
        const auto geometry = geometryList[0];
    }

    CurvedDetectorDescriptor::CurvedDetectorDescriptor(const IndexVector_t& numOfCoeffsPerDim,
                                                       const RealVector_t& spacingPerDim,
                                                       const std::vector<Geometry>& geometryList,
                                                       real_t scale)
        : DetectorDescriptor(numOfCoeffsPerDim, spacingPerDim, geometryList), _scale(scale)
    {
    }

    DetectorDescriptor::Ray
        CurvedDetectorDescriptor::computeRayFromDetectorCoord(const RealVector_t& detectorCoord,
                                                              const index_t poseIndex) const
    {
        // Assume detectorCoord is 2D describing a pixel on the detector
        // To transform it into a global coord, we need to create cylindrical coordinates from
        // it, we have to figure out the z (so "up/down" direction)
        // And then x and y are determined by an angle phi, which describes the "fan" angle from the
        // principal point to the point in x and y on the plane z = 0

        assert(std::make_unsigned_t<index_t>(poseIndex) < _geometry.size());

        // Scale detector coords to range [0, 1]
        const RealVector_t pixel = detectorCoord.template cast<real_t>().array()
                                   / getNumberOfCoefficientsPerDimension()
                                         .head(getNumberOfDimensions() - 1)
                                         .template cast<real_t>()
                                         .array();

        const auto geometry = _geometry[poseIndex];

        const RealVector_t ro = geometry.getCameraCenter();
        const RealVector_t rd = mapCurvedPixelToDir(pixel, geometry);

        return Ray(ro, rd);
    }

    RealVector_t CurvedDetectorDescriptor::mapCurvedPixelToDir(const RealVector_t& p,
                                                               const Geometry& geom) const
    {
        // Calculate distance from source to principal point
        const auto source = geom.getCameraCenter();
        const auto pp = geom.getPrincipalPoint();

        const auto theta = geom.getFanAngle();

        const auto radius = _scale * (pp - source).norm(); // TODO scale it correctly

        // Subtract scaled y-value from 1, to flip it, so a y-value of 0, is at the top and a
        // y-value of 1, at the bottom Then shift it by 0.5, so pixel value of 0.5 is at z == 0
        const auto z = [&]() -> real_t {
            if (getNumberOfDimensions() == 3) {
                return (1.f - p[1]) - 0.5f;
            } else {
                return 0;
            }
        }();

        // Phi is the angle between the principal point and the pixel
        const auto phi = geometry::Radian{(p[0] * theta * 2) - theta};

        // Local is equal for 2D and 3D, except 3D is also using the z coordinate, in 2D that's just
        // ignored
        const RealVector_t local = [&]() {
            RealVector_t local(getNumberOfDimensions());
            const auto x = radius * std::cos(phi);
            const auto y = radius * std::sin(phi);

            if (getNumberOfDimensions() == 3) {
                // Coordinates are fliped according to our needs, as x in cylindrical is y in
                // world/volume coordinates y in cylindrical is z in world/volume coordinates, and z
                // in cylindrical is x in world/volume coords
                local << z, x, y;
            } else if (getNumberOfDimensions() == 2) {
                // x and y coord are fliped compared to the standard definition of cylindrical
                // coordinates
                local << y, x;
            }
            return local;
        }();

        // Rotate point, by the inverse (== transpose for rotation matrices)
        auto rot = geom.getInverseRotationMatrix();

        return (rot * local).normalized();
    }

    RealVector_t
        CurvedDetectorDescriptor::computeDetectorCoordFromRay(const Ray& /* ray, */,
                                                              const index_t /* poseIndex */) const
    {
        return RealVector_t(2);
    }

    bool CurvedDetectorDescriptor::isEqual(const DataDescriptor& other) const
    {
        // CurvedDetectorDescriptor has no data, so just deligate it to base class
        return DetectorDescriptor::isEqual(other);
    }

    CurvedDetectorDescriptor* CurvedDetectorDescriptor::cloneImpl() const
    {
        return new CurvedDetectorDescriptor(getNumberOfCoefficientsPerDimension(),
                                            getSpacingPerDimension(), _geometry, _theta);
    }
} // namespace elsa
