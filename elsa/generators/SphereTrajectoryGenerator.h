#pragma once

#include "Geometry.h"
#include "DetectorDescriptor.h"
#include "TrajectoryGenerator.h"

#include <vector>
#include <utility>

namespace elsa
{
    /**
     * @brief Generator for spherical trajectories as used in X-ray Computed Tomography
     * (for 2d/3d).
     *
     * @author Michael Loipführer - initial code
     */

    class SphereTrajectoryGenerator : public TrajectoryGenerator
    {
    public:
        /**
         * @brief Generate a list of geometries corresponding to a spherical trajectory around a
         * volume. The spherical trajectory is made up of multiple circular trajectories around
         * the volume.
         *
         * @param numberOfPoses the number of (equally spaced) acquisition poses to be generated
         * @param volumeDescriptor the volume around which the trajectory should go
         * @param numberOfCircles the number of circular trajectories this acquisition path is made
         * up of
         * @param sourceToCenter the distance of the X-ray source to
         * the center of the volume
         * @param centerToDetector the distance of the center of the volume
         * to the X-ray detector
         *
         * @returns a DetectorDescriptor describing the spherical trajectory
         *
         * Please note: the sinogram size/spacing will match the volume size/spacing.
         *
         * TODO: Make it possible to return either PlanarDetectorDescriptor, or
         * CurvedDetectorDescriptor
         */
        static std::unique_ptr<DetectorDescriptor>
            createTrajectory(index_t numberOfPoses, const DataDescriptor& volumeDescriptor,
                             index_t numberOfCircles,
                             geometry::SourceToCenterOfRotation sourceToCenter,
                             geometry::CenterOfRotationToDetector centerToDetector);
    };
} // namespace elsa
