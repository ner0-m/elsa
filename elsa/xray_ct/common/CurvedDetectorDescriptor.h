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
     * @details The idea behind the current approach is to reduce the problem of computing
     * intersections between rays and a curved detector to the planar detector case. We project
     * coordinates of the curved detector onto coordinates on a virtual flat detector behind.
     * Conceptually, the flat detector has the same amount of pixels but they become increasingly
     * narrower towards the endpoints of the flat detector. The coordinates for the principal ray
     * are the same for the flat and the curved detector.
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
         *
         * @details The information needed to model a curved detector includes the information
         * required for a planar detector. Additionally, we need a parameter describing the
         * curvature. Currently, this is implemented by providing the fanout angle (in radians),
         * which we use internally to compute the radius of the curved detector. Furthermore, we
         * need a parameter for the distance of the source to the detector (so, the sum of the
         * distances "source to center" and "center to detector").
         */
        CurvedDetectorDescriptor(const IndexVector_t& numOfCoeffsPerDim,
                                 const RealVector_t& spacingPerDim,
                                 const std::vector<Geometry>& geometryList,
                                 const geometry::Radian angle, const real_t s2d);

        /**
         * @brief Construct a CurvedDetectorDescriptor with given number of coefficients
         * per dimension, a list of geometry poses in the trajectory, an angle in radians, and
         * the length from the source to the detector.
         *
         * @details The information needed to model a curved detector includes the information
         * required for a planar detector. Additionally, we need a parameter describing the
         * curvature. Currently, this is implemented by providing the fanout angle (in radians),
         * which we use internally to compute the radius of the curved detector. Furthermore, we
         * need a parameter for the distance of the source to the detector (so, the sum of the
         * distances "source to center" and "center to detector").
         */
        CurvedDetectorDescriptor(const IndexVector_t& numOfCoeffsPerDim,
                                 const std::vector<Geometry>& geometryList,
                                 const geometry::Radian angle, const real_t s2d);

        using DetectorDescriptor::computeRayFromDetectorCoord;

        /**
         * @details The ray computation of the curved detector descriptor is a wrapper around the
         * ray computation of the planar detector descriptor. The wrapper is responsible for mapping
         * the user-provided coordinates of the curved detector to the corresponding coordinates of
         * the virtual planar detector descriptor. This overhead is encapsulated in the
         * mapCurvedCoordToPlanarCoord method, which "extends" the rays hitting the curved
         * detector to the virtual flat detector behind.  Most importantly, the
         * CurvedDetectorDescriptor class overrides the computeRayFromDetectorCoord function.
         * This function receives the parameter detectorCoord which is treated as the index of a
         * pixel on the curved detector. If the x-coordinate of detectorCoord is of the form x.5,
         * i.e.  references the pixel center, we can use the coordinates that were precomputed in
         * the constructor.  This should usually be the case. Otherwise, the planar detector
         * coordinate needs to be dynamically computed by mapCurvedCoordToPlanarCoord. Finally, we
         * receive a coordinate in the detector space of the flat detector which we pass to the
         * parent implementation of computeRayFromDetectorCoord.
         *
         * ![Schematic Overview](docs/curved-detector-schematic-overview.png)
         */
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
