#include "CurvedCircleTrajectoryGenerator.h"
#include "CurvedDetectorDescriptor.h"

#include <optional>

namespace elsa
{
    std::unique_ptr<CurvedDetectorDescriptor> CurvedCircleTrajectoryGenerator::createTrajectory(
        index_t numberOfPoses, const DataDescriptor& volumeDescriptor, index_t arcDegrees,
        real_t sourceToCenter, real_t centerToDetector, geometry::Radian angle,
        std::optional<RealVector_t> principalPointOffset,
        std::optional<RealVector_t> centerOfRotOffset, std::optional<IndexVector_t> detectorSize,
        std::optional<RealVector_t> detectorSpacing)
    {
        auto [coeffs, spacing, geometryList] = BaseCircleTrajectoryGenerator::createTrajectoryData(
            numberOfPoses, volumeDescriptor, arcDegrees, sourceToCenter, centerToDetector,
            principalPointOffset, centerOfRotOffset, detectorSize, detectorSpacing);

        return std::make_unique<CurvedDetectorDescriptor>(std::move(coeffs), std::move(spacing),
                                                          std::move(geometryList), angle,
                                                          sourceToCenter + centerToDetector);
    }
} // namespace elsa
