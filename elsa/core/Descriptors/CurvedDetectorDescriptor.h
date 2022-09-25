#pragma once

#include "elsaDefines.h"
#include "DetectorDescriptor.h"
#include "StrongTypes.h"

#include <tuple>
#include <vector>

namespace elsa
{
    /**
     * @brief Class representing a curved detector surface. It uses a virtual
     * PlanarDetectorDescriptor in the background by mapping the coordinates of the curved detector
     * detector to coordinates of the planar one.
     *
     * @author Julia Spindler, Robert Imschweiler - adapt PlanarDetectorDescriptor for
     * CurvedDetectorDescriptor
     */
    class CurvedDetectorDescriptor : public DetectorDescriptor
    {
    public:
        CurvedDetectorDescriptor() = delete;

        ~CurvedDetectorDescriptor() = default;

        /**
         * @brief Construct a CurvedDetectorDescriptor with given number of coefficients and spacing
         * per dimension, a list of geometry poses in the trajectory, an angle in radians, and
         * the length from the source to the detector.
         */
        CurvedDetectorDescriptor(const IndexVector_t& numOfCoeffsPerDim,
                                 const RealVector_t& spacingPerDim,
                                 const std::vector<Geometry>& geometryList,
                                 const geometry::Radian angle, const real_t s2d);

        /**
         * @brief Construct a CurvedDetectorDescriptor with given number of coefficients
         * per dimension, a list of geometry poses in the trajectory, an angle in radians, and
         * the length from the source to the detector.
         */
        CurvedDetectorDescriptor(const IndexVector_t& numOfCoeffsPerDim,
                                 const std::vector<Geometry>& geometryList,
                                 const geometry::Radian angle, const real_t s2d);

        using DetectorDescriptor::computeRayFromDetectorCoord;

        RealRay_t computeRayFromDetectorCoord(const RealVector_t& detectorCoord,
                                              const index_t poseIndex) const override;

        /**
         * @brief Return the coordinates of the planar detector descriptor which
         * operates in background.
         */
        const std::vector<RealVector_t>& getPlanarCoords() const;

        real_t getRadius() const;

    private:
        CurvedDetectorDescriptor* cloneImpl() const override;

        bool isEqual(const DataDescriptor& other) const override;

        /**
         * @brief setup function that is called in the constructor.
         * Precomputes the coordinates of the pixel midpoints on the hypothetical planar detector
         */
        void setup();

        /**
         * @brief Map a given coordinate of the curved detector to the corresponding coordinate of
         * the virtual flat detector.
         *
         * @param coord
         * @return real_t
         */
        RealVector_t mapCurvedCoordToPlanarCoord(const RealVector_t& coord) const;

        // the coordinates of the pixel midpoints on the flat detector
        std::vector<RealVector_t> _planarCoords;

        // angle describing the fov of the detector
        geometry::Radian _angle;

        // radius of the circle the curved detector is a part of
        real_t _radius;

        // the detector pixels are evenly spaced, so the angle distance between them is fov/detector
        // length
        real_t _angleDelta;

        // the middle coordinate that the principal ray hits
        RealVector_t _detectorCenter;

        // the distance from the source to the detector
        real_t _s2d;

        // the center of the circle the curved detector lies on
        RealVector_t _centerOfCircle;
    };
} // namespace elsa
