#include "CircleTrajectoryGenerator.h"
#include "PlanarDetectorDescriptor.h"

#include <optional>
#include <vector>

namespace elsa
{
    std::unique_ptr<PlanarDetectorDescriptor> CircleTrajectoryGenerator::createTrajectory(
        index_t numberOfPoses, const DataDescriptor& volumeDescriptor, index_t arcDegrees,
        real_t sourceToCenter, real_t centerToDetector,
        std::optional<RealVector_t> principalPointOffset,
        std::optional<RealVector_t> centerOfRotOffset, std::optional<IndexVector_t> detectorSize,
        std::optional<RealVector_t> detectorSpacing)
    {
        auto thetas = [&]() {
            auto tmp = RealVector_t::LinSpaced(numberOfPoses, 0, static_cast<real_t>(arcDegrees));
            return std::vector<real_t>{tmp.begin(), tmp.end()};
        }();

        return trajectoryFromAngles(thetas, volumeDescriptor, sourceToCenter, centerToDetector,
                                    principalPointOffset, centerOfRotOffset, detectorSize,
                                    detectorSpacing);
    }

    std::unique_ptr<PlanarDetectorDescriptor> CircleTrajectoryGenerator::trajectoryFromAngles(
        const std::vector<real_t>& thetas, const DataDescriptor& volumeDescriptor,
        real_t sourceToCenter, real_t centerToDetector,
        std::optional<RealVector_t> principalPointOffset,
        std::optional<RealVector_t> centerOfRotOffset, std::optional<IndexVector_t> detectorSize,
        std::optional<RealVector_t> detectorSpacing)
    {
        auto [coeffs, spacing, geometryList] = BaseCircleTrajectoryGenerator::createTrajectoryData(
            thetas, volumeDescriptor, sourceToCenter, centerToDetector, principalPointOffset,
            centerOfRotOffset, detectorSize, detectorSpacing);

        return std::make_unique<PlanarDetectorDescriptor>(std::move(coeffs), std::move(spacing),
                                                          std::move(geometryList));
    }
} // namespace elsa
