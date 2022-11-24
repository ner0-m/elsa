#include "BaseCircleTrajectoryGenerator.h"
#include "Logger.h"
#include "VolumeDescriptor.h"

#include <numeric>
#include <optional>
#include <stdexcept>

#include "spdlog/fmt/fmt.h"
#include "spdlog/fmt/ranges.h"

namespace elsa
{

    std::tuple<IndexVector_t, RealVector_t, std::vector<Geometry>>
        BaseCircleTrajectoryGenerator::createTrajectoryData(
            index_t numberOfPoses, const DataDescriptor& volumeDescriptor, index_t arcDegrees,
            real_t sourceToCenter, real_t centerToDetector,
            std::optional<RealVector_t> principalPointOffset,
            std::optional<RealVector_t> centerOfRotOffset,
            std::optional<IndexVector_t> detectorSize, std::optional<RealVector_t> detectorSpacing)
    {
        // pull in geometry namespace, to reduce cluttering
        using namespace geometry;

        // sanity check
        const auto dim = volumeDescriptor.getNumberOfDimensions();

        if (dim < 2 || dim > 3)
            throw InvalidArgumentError("CircleTrajectoryGenerator: can only handle 2d/3d");

        Logger::get("CircleTrajectoryGenerator")
            ->info("creating {}D trajectory with {} poses in an {} degree arc", dim, numberOfPoses,
                   arcDegrees);

        // Calculate size and spacing for each geometry pose using a IIFE
        const auto [coeffs, spacing] = calculateSizeAndSpacingPerGeometry(
            volumeDescriptor, numberOfPoses, detectorSize, detectorSpacing);

        // Create vector and reserve the necessary size, minor optimization such that no new
        // allocations are necessary in the loop
        std::vector<Geometry> geometryList;
        geometryList.reserve(static_cast<std::size_t>(numberOfPoses));

        auto degrees = RealVector_t::LinSpaced(numberOfPoses, 0, static_cast<real_t>(arcDegrees));

        for (auto degree : degrees) {
            const auto angle = Degree{degree}.to_radian();
            Logger::get("CircleTrajectoryGenerator")
                ->info("Using angle {}", static_cast<real_t>(degree));
            if (dim == 2) {
                // Use emplace_back, then no copy is created
                geometryList.emplace_back(
                    SourceToCenterOfRotation{sourceToCenter},
                    CenterOfRotationToDetector{centerToDetector}, Radian{angle},
                    VolumeData2D{volumeDescriptor.getSpacingPerDimension(),
                                 volumeDescriptor.getLocationOfOrigin()},
                    SinogramData2D{Size2D{coeffs}, Spacing2D{spacing}},
                    principalPointOffset ? PrincipalPointOffset{principalPointOffset.value()[0]}
                                         : PrincipalPointOffset{0},
                    centerOfRotOffset ? RotationOffset2D{centerOfRotOffset.value()}
                                      : RotationOffset2D{0, 0});
            } else {
                geometryList.emplace_back(
                    SourceToCenterOfRotation{sourceToCenter},
                    CenterOfRotationToDetector{centerToDetector},
                    VolumeData3D{volumeDescriptor.getSpacingPerDimension(),
                                 volumeDescriptor.getLocationOfOrigin()},
                    SinogramData3D{Size3D{coeffs}, Spacing3D{spacing}},
                    RotationAngles3D{Radian{angle}, Radian{0}, Radian{0}},
                    principalPointOffset ? PrincipalPointOffset2D{principalPointOffset.value()}
                                         : PrincipalPointOffset2D{0, 0},
                    centerOfRotOffset ? RotationOffset3D{centerOfRotOffset.value()}
                                      : RotationOffset3D{0, 0, 0});
            }
        }
        Logger::get("CircleTrajectoryGenerator")
            ->info("size of geometry {}, coeffs: {}, {}", geometryList.size(), coeffs[0],
                   coeffs[1]);

        return std::make_tuple(coeffs, spacing, geometryList);
    }

    std::tuple<IndexVector_t, RealVector_t, std::vector<Geometry>>
        BaseCircleTrajectoryGenerator::fromAngularIncrement(
            index_t numberOfPoses, const DataDescriptor& volumeDescriptor, real_t angularInc,
            real_t sourceToCenter, real_t centerToDetector,
            std::optional<RealVector_t> principalPointOffset,
            std::optional<RealVector_t> centerOfRotOffset,
            std::optional<IndexVector_t> detectorSize, std::optional<RealVector_t> detectorSpacing)
    {
        // pull in geometry namespace, to reduce cluttering
        using namespace geometry;

        // sanity check
        const auto dim = volumeDescriptor.getNumberOfDimensions();

        if (dim < 2 || dim > 3)
            throw InvalidArgumentError("CircleTrajectoryGenerator: can only handle 2d/3d");

        Logger::get("CircleTrajectoryGenerator")
            ->info("creating {}D trajectory with {} poses with an angular increment of {}", dim,
                   numberOfPoses, angularInc);

        // Calculate size and spacing for each geometry pose using a IIFE
        const auto [coeffs, spacing] = calculateSizeAndSpacingPerGeometry(
            volumeDescriptor, numberOfPoses, detectorSize, detectorSpacing);

        // Create vector and reserve the necessary size, minor optimization such that no new
        // allocations are necessary in the loop
        std::vector<Geometry> geometryList;
        geometryList.reserve(static_cast<std::size_t>(numberOfPoses));

        std::vector<real_t> degrees(numberOfPoses);
        std::iota(degrees.begin(), degrees.end(), 0);
        std::transform(degrees.begin(), degrees.end(), degrees.begin(),
                       [=](auto idx) { return static_cast<real_t>(angularInc) * idx; });

        for (auto degree : degrees) {
            const auto angle = Degree{degree}.to_radian();
            if (dim == 2) {
                // Use emplace_back, then no copy is created
                geometryList.emplace_back(
                    SourceToCenterOfRotation{sourceToCenter},
                    CenterOfRotationToDetector{centerToDetector}, Radian{angle},
                    VolumeData2D{volumeDescriptor.getSpacingPerDimension(),
                                 volumeDescriptor.getLocationOfOrigin()},
                    SinogramData2D{Size2D{coeffs}, Spacing2D{spacing}},
                    principalPointOffset ? PrincipalPointOffset{principalPointOffset.value()[0]}
                                         : PrincipalPointOffset{0},
                    centerOfRotOffset ? RotationOffset2D{centerOfRotOffset.value()}
                                      : RotationOffset2D{0, 0});
            } else {
                geometryList.emplace_back(
                    SourceToCenterOfRotation{sourceToCenter},
                    CenterOfRotationToDetector{centerToDetector},
                    VolumeData3D{volumeDescriptor.getSpacingPerDimension(),
                                 volumeDescriptor.getLocationOfOrigin()},
                    SinogramData3D{Size3D{coeffs}, Spacing3D{spacing}},
                    RotationAngles3D{Radian{angle}, Radian{0}, Radian{0}},
                    principalPointOffset ? PrincipalPointOffset2D{principalPointOffset.value()}
                                         : PrincipalPointOffset2D{0, 0},
                    centerOfRotOffset ? RotationOffset3D{centerOfRotOffset.value()}
                                      : RotationOffset3D{0, 0, 0});
            }
        }

        return std::make_tuple(coeffs, spacing, geometryList);
    }
} // namespace elsa
