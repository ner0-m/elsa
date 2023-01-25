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
        auto thetas = [&]() {
            auto tmp = RealVector_t::LinSpaced(numberOfPoses, 0, static_cast<real_t>(arcDegrees));
            return std::vector<real_t>{tmp.begin(), tmp.end()};
        }();

        auto [coeffs, spacing, geometryList] = BaseCircleTrajectoryGenerator::createTrajectoryData(
            thetas, volumeDescriptor, sourceToCenter, centerToDetector, principalPointOffset,
            centerOfRotOffset, detectorSize, detectorSpacing);

        return std::make_unique<CurvedDetectorDescriptor>(std::move(coeffs), std::move(spacing),
                                                          std::move(geometryList), angle,
                                                          sourceToCenter + centerToDetector);
    }
} // namespace elsa
