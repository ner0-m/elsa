#include "CircleTrajectoryGenerator.h"
#include "PlanarDetectorDescriptor.h"

#include <optional>

namespace elsa
{
    std::unique_ptr<PlanarDetectorDescriptor> CircleTrajectoryGenerator::createTrajectory(
        index_t numberOfPoses, const DataDescriptor& volumeDescriptor, index_t arcDegrees,
        real_t sourceToCenter, real_t centerToDetector,
        std::optional<RealVector_t> principalPointOffset,
        std::optional<RealVector_t> centerOfRotOffset, std::optional<IndexVector_t> detectorSize,
        std::optional<RealVector_t> detectorSpacing)
    {
        auto [coeffs, spacing, geometryList] = BaseCircleTrajectoryGenerator::createTrajectoryData(
            numberOfPoses, volumeDescriptor, arcDegrees, sourceToCenter, centerToDetector,
            principalPointOffset, centerOfRotOffset, detectorSize, detectorSpacing);

        return std::make_unique<PlanarDetectorDescriptor>(std::move(coeffs), std::move(spacing),
                                                          std::move(geometryList));
    }

    std::unique_ptr<PlanarDetectorDescriptor> CircleTrajectoryGenerator::fromAngularIncrement(
        index_t numberOfPoses, const DataDescriptor& volumeDescriptor, real_t angularInc,
        real_t sourceToCenter, real_t centerToDetector,
        std::optional<RealVector_t> principalPointOffset,
        std::optional<RealVector_t> centerOfRotOffset, std::optional<IndexVector_t> detectorSize,
        std::optional<RealVector_t> detectorSpacing)
    {
        auto [coeffs, spacing, geometryList] = BaseCircleTrajectoryGenerator::fromAngularIncrement(
            numberOfPoses, volumeDescriptor, angularInc, sourceToCenter, centerToDetector,
            principalPointOffset, centerOfRotOffset, detectorSize, detectorSpacing);

        return std::make_unique<PlanarDetectorDescriptor>(coeffs, spacing, geometryList);
    }
} // namespace elsa
