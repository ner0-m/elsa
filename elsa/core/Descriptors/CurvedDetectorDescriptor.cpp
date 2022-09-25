#include "CurvedDetectorDescriptor.h"
#include "TypeCasts.hpp"

#include <vector>

namespace elsa
{
    CurvedDetectorDescriptor::CurvedDetectorDescriptor(const IndexVector_t& numOfCoeffsPerDim,
                                                       const RealVector_t& spacingPerDim,
                                                       const std::vector<Geometry>& geometryList,
                                                       const geometry::Radian angle,
                                                       const real_t s2d)
        : DetectorDescriptor(numOfCoeffsPerDim, spacingPerDim, geometryList),
          _angle{angle},
          _radius{static_cast<real_t>(numOfCoeffsPerDim[0]) / static_cast<real_t>(angle)},
          _angleDelta{_angle / static_cast<real_t>(numOfCoeffsPerDim[0])},
          _s2d{s2d}
    {
        setup();
    }
    CurvedDetectorDescriptor::CurvedDetectorDescriptor(const IndexVector_t& numOfCoeffsPerDim,
                                                       const std::vector<Geometry>& geometryList,
                                                       const geometry::Radian angle,
                                                       const real_t s2d)
        : DetectorDescriptor(numOfCoeffsPerDim, geometryList),
          _angle{angle},
          _radius{static_cast<real_t>(numOfCoeffsPerDim[0]) / static_cast<real_t>(angle)},
          _angleDelta{_angle / static_cast<real_t>(numOfCoeffsPerDim[0])},
          _s2d{s2d}
    {
        setup();
    }

    void CurvedDetectorDescriptor::setup()
    {
        index_t detectorLen{_numberOfCoefficientsPerDimension[0]};
        index_t detectorHeight{_numberOfDimensions == 3 ? _numberOfCoefficientsPerDimension[1] : 1};

        _detectorCenter = RealVector_t(2);
        _detectorCenter << static_cast<real_t>(detectorLen) / static_cast<real_t>(2.0),
            static_cast<real_t>(detectorHeight) / static_cast<real_t>(2.0);

        _centerOfCircle = RealVector_t(2);
        _centerOfCircle << _detectorCenter[0], _s2d - _radius;

        _planarCoords.reserve(detectorLen * detectorHeight);

        for (index_t coordH{0}; coordH < detectorHeight; coordH++) {
            for (index_t coordL{0}; coordL < detectorLen; coordL++) {
                // shift by 0.5 to get the midpoint of the detector pixel
                RealVector_t pixelCenter(_numberOfDimensions - 1);
                if (_numberOfDimensions == 2) {
                    pixelCenter << static_cast<real_t>(coordL) + static_cast<real_t>(0.5);
                } else {
                    pixelCenter << static_cast<real_t>(coordL) + static_cast<real_t>(0.5),
                        static_cast<real_t>(coordH) + static_cast<real_t>(0.5);
                }
                _planarCoords.push_back(mapCurvedCoordToPlanarCoord(pixelCenter));
            }
        }
    }

    inline RealVector_t
        CurvedDetectorDescriptor::mapCurvedCoordToPlanarCoord(const RealVector_t& coord) const
    {
        // angle between the principal ray and the current ray we want to compute
        real_t rayAngle{(coord[0] - _detectorCenter[0]) * _angleDelta};

        // deltas between the center and the point on the curved detector
        real_t delta_x{static_cast<real_t>(sin(rayAngle)) * _radius};
        real_t delta_depth{static_cast<real_t>(cos(rayAngle)) * _radius};

        // the coordinate of the point on the curved detector in detector space
        RealVector_t pointOnCurved(2);
        pointOnCurved << _centerOfCircle[0] + delta_x, _centerOfCircle[1] + delta_depth;
        // if the curvature is very small and the points are on top of each other, the factor is
        // zero
        real_t factor{pointOnCurved[1] == 0 ? 1 : _s2d / pointOnCurved[1]};

        // return new coordinate by scaling the vector onto the flat detector
        RealVector_t pointOnFlat(_numberOfDimensions - 1);
        if (_numberOfDimensions == 3) {
            pointOnFlat << (pointOnCurved[0] - _detectorCenter[0]) * factor + _detectorCenter[0],
                (coord[1] - _detectorCenter[1]) * factor + _detectorCenter[1];
        } else {
            pointOnFlat << (pointOnCurved[0] - _detectorCenter[0]) * factor + _detectorCenter[0];
        }

        return pointOnFlat;
    }

    RealRay_t
        CurvedDetectorDescriptor::computeRayFromDetectorCoord(const RealVector_t& detectorCoord,
                                                              const index_t poseIndex) const
    {
        index_t pixelIndex{static_cast<index_t>(detectorCoord[0])};

        // instead of multiplying the detectorCoord with the projection matrix we retrieve the
        // actual coordinate in detector space by treating detectorCoord as an Index into
        // _planarCoords
        bool isPixelCenter{
            std::abs(detectorCoord[0] - static_cast<real_t>(pixelIndex) - static_cast<real_t>(0.5))
            <= static_cast<real_t>(0.0001)};

        if (_numberOfDimensions == 3) {
            index_t pixelIndexWidth{static_cast<index_t>(detectorCoord[1])};
            isPixelCenter = isPixelCenter
                            && std::abs(detectorCoord[1] - static_cast<real_t>(pixelIndexWidth)
                                        - static_cast<real_t>(0.5))
                                   <= static_cast<real_t>(0.0001);
            pixelIndex += pixelIndexWidth * _numberOfCoefficientsPerDimension[0];
        }

        RealVector_t curvedDetectorCoord{
            isPixelCenter ? _planarCoords[pixelIndex] : mapCurvedCoordToPlanarCoord(detectorCoord)};

        return DetectorDescriptor::computeRayFromDetectorCoord(curvedDetectorCoord, poseIndex);
    }

    bool CurvedDetectorDescriptor::isEqual(const DataDescriptor& other) const
    {
        const CurvedDetectorDescriptor* otherPtr{
            downcast_safe<const CurvedDetectorDescriptor, const DataDescriptor>(&other)};

        return (otherPtr != nullptr && _angle == otherPtr->_angle && _radius == otherPtr->_radius
                && _s2d == otherPtr->_s2d && DetectorDescriptor::isEqual(other));
    }

    CurvedDetectorDescriptor* CurvedDetectorDescriptor::cloneImpl() const
    {
        return new CurvedDetectorDescriptor(getNumberOfCoefficientsPerDimension(),
                                            getSpacingPerDimension(), _geometry, _angle, _s2d);
    }

    const std::vector<RealVector_t>& CurvedDetectorDescriptor::getPlanarCoords() const
    {
        return _planarCoords;
    }

    real_t CurvedDetectorDescriptor::getRadius() const
    {
        return _radius;
    }
} // namespace elsa
