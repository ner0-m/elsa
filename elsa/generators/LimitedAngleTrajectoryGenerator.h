#pragma once

#include "Geometry.h"
#include "DetectorDescriptor.h"

#include <vector>
#include <utility>

namespace elsa
{
    /**
     * @brief Generator for limited angle trajectories as used in X-ray Computed Tomography
     * (for 2D/3D).
     *
     * @author Andi Braimllari - initial code
     */
    class LimitedAngleTrajectoryGenerator
    {
    public:
        /**
         * @brief Generate a list of geometries corresponding to a limited angle trajectory around a
         * volume.
         *
         * @param numberOfPoses the number of (equally spaced) acquisition poses to be generated
         * @param missingWedgeAngles the angles between which the missing wedge is located
         * @param volumeDescriptor the volume around which the trajectory should go
         * @param arcDegrees the size of the arc of the circle covered by the trajectory (in
         * degrees, 360 for full circle)
         * @param sourceToCenter the distance of the X-ray source to
         * the center of the volume
         * @param centerToDetector the distance of the center of the volume
         * to the X-ray detector
         *
         * @returns a pair containing the list of geometries with a circular trajectory, and the
         * sinogram data descriptor
         *
         * Please note: the first pose will be at 0 degrees, the last pose will be at arcDegrees
         * For example: 3 poses over a 180 arc will yield: 0, 90, 180 degrees.
         *
         * Please note: the sinogram size/spacing will match the volume size/spacing.
         *
         * TODO: Make it possible to return either PlanarDetectorDescriptor, or
         *  CurvedDetectorDescriptor
         */
        static std::unique_ptr<DetectorDescriptor>
            createTrajectory(index_t numberOfPoses, RealVector_t missingWedgeAngles,
                             const DataDescriptor& volumeDescriptor, index_t arcDegrees,
                             real_t sourceToCenter, real_t centerToDetector);
    };
} // namespace elsa
