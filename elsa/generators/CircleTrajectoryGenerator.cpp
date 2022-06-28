#include "CircleTrajectoryGenerator.h"
#include "Logger.h"
#include "VolumeDescriptor.h"
#include "PlanarDetectorDescriptor.h"

#include <stdexcept>

namespace elsa
{
    std::unique_ptr<DetectorDescriptor> CircleTrajectoryGenerator::createTrajectory(
        index_t numberOfPoses, const DataDescriptor& volumeDescriptor, index_t arcDegrees,
        real_t sourceToCenter, real_t centerToDetector)
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
        const auto [coeffs, spacing] =
            calculateSizeAndSpacingPerGeometry(volumeDescriptor, numberOfPoses);

        // Create vector and reserve the necessary size, minor optimization such that no new
        // allocations are necessary in the loop
        std::vector<Geometry> geometryList;
        geometryList.reserve(static_cast<std::size_t>(numberOfPoses));

        for (auto degree : RealVector_t::LinSpaced(numberOfPoses, 0, arcDegrees)) {
            const auto angle = Degree{degree}.to_radian();
            if (dim == 2) {
                // Use emplace_back, then no copy is created
                geometryList.emplace_back(SourceToCenterOfRotation{sourceToCenter},
                                          CenterOfRotationToDetector{centerToDetector},
                                          Radian{angle},
                                          VolumeData2D{volumeDescriptor.getSpacingPerDimension(),
                                                       volumeDescriptor.getLocationOfOrigin()},
                                          SinogramData2D{Size2D{coeffs}, Spacing2D{spacing}});
            } else {
                geometryList.emplace_back(SourceToCenterOfRotation{sourceToCenter},
                                          CenterOfRotationToDetector{centerToDetector},
                                          VolumeData3D{volumeDescriptor.getSpacingPerDimension(),
                                                       volumeDescriptor.getLocationOfOrigin()},
                                          SinogramData3D{Size3D{coeffs}, Spacing3D{spacing}},
                                          RotationAngles3D{Radian{angle}, Radian{0}, Radian{0}});
            }
        }

        return std::make_unique<PlanarDetectorDescriptor>(coeffs, spacing, geometryList);
    }

} // namespace elsa
