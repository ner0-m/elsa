#pragma once

#include "Geometry.h"

#include "DetectorDescriptor.h"
#include "TrajectoryGenerator.h"

#include <optional>
#include <vector>
#include <utility>

namespace elsa
{
    /**
     * @brief Generator for traditional circular trajectories as used in X-ray Computed Tomography
     * (for 2d/3d).
     *
     * @author Maximilan Hornung - initial code
     * @author Tobias Lasser - modernization, fixes
     * @author Julia Spindler, Robert Imschweiler - refactor
     */
    class BaseCircleTrajectoryGenerator : public TrajectoryGenerator
    {
        /**
         * @brief Generate a list of geometries corresponding to a circular trajectory around a
         * volume.
         *
         * @param numberOfPoses the number of (equally spaced) acquisition poses to be generated
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
         */
    protected:
        static std::tuple<IndexVector_t, RealVector_t, std::vector<Geometry>>
            createTrajectoryData(index_t numberOfPoses, const DataDescriptor& volumeDescriptor,
                                 index_t arcDegrees, real_t sourceToCenter, real_t centerToDetector,
                                 std::optional<RealVector_t> principalPointOffset = std::nullopt,
                                 std::optional<RealVector_t> centerOfRotOffset = std::nullopt,
                                 std::optional<IndexVector_t> detectorSize = std::nullopt,
                                 std::optional<RealVector_t> detectorSpacing = std::nullopt);
    };
} // namespace elsa
