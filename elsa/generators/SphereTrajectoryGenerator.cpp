#include "SphereTrajectoryGenerator.h"
#include "PlanarDetectorDescriptor.h"
#include "Logger.h"

namespace elsa
{
    std::unique_ptr<DetectorDescriptor> SphereTrajectoryGenerator::createTrajectory(
        index_t numberOfPoses, const DataDescriptor& volumeDescriptor, index_t numberOfCircles,
        geometry::SourceToCenterOfRotation sourceToCenter,
        geometry::CenterOfRotationToDetector centerToDetector)
    {
        // pull in geometry namespace, to reduce cluttering
        using namespace geometry;

        // sanity check
        const auto dim = volumeDescriptor.getNumberOfDimensions();

        if (dim != 3)
            throw InvalidArgumentError("SphereTrajectoryGenerator: can only handle 3d");

        Logger::get("SphereTrajectoryGenerator")
            ->info("creating {}D trajectory with {} poses consisting of {} circular "
                   "trajectories",
                   dim, numberOfPoses, numberOfCircles);

        const auto [coeffs, spacing] =
            calculateSizeAndSpacingPerGeometry(volumeDescriptor, numberOfPoses);

        // Create vector and reserve the necessary size, minor optimization such that no new
        // allocations are necessary in the loop
        std::vector<Geometry> geometryList;
        geometryList.reserve(static_cast<std::size_t>(numberOfPoses));

        // compute the total circumference of all circular trajectories we are generating
        double circumference = 0;
        for (index_t i = 0; i < numberOfCircles; ++i) {
            circumference +=
                M_PI * 2.0 * centerToDetector
                * cos(M_PI * static_cast<double>(i + 1) / static_cast<double>(numberOfCircles + 2)
                      - M_PI_2);
        }

        auto circumferenceChangePerPose = circumference / static_cast<double>(numberOfPoses);
        index_t createdPoses = 0;

        for (index_t i = 0; i < numberOfCircles; ++i) {
            auto beta = M_PI * static_cast<double>(i + 1) / static_cast<double>(numberOfCircles + 2)
                        - M_PI_2;

            index_t posesInCircle;
            if (i == numberOfCircles - 1) {
                posesInCircle = numberOfPoses - createdPoses;
            } else {
                posesInCircle = static_cast<index_t>(
                    (2.0 * M_PI * static_cast<double>(centerToDetector) * cos(beta))
                    / circumferenceChangePerPose);
            }

            for (index_t poseIndex = 0; poseIndex < posesInCircle; ++poseIndex) {
                auto gamma = M_PI * 2.0 * static_cast<double>(poseIndex)
                             / static_cast<double>(posesInCircle);

                geometryList.emplace_back(sourceToCenter, centerToDetector,
                                          VolumeData3D{volumeDescriptor.getSpacingPerDimension(),
                                                       volumeDescriptor.getLocationOfOrigin()},
                                          SinogramData3D{Size3D{coeffs}, Spacing3D{spacing}},
                                          RotationAngles3D{Radian{static_cast<real_t>(gamma)},
                                                           Radian{static_cast<real_t>(beta)},
                                                           Radian{0}});
            }

            createdPoses += posesInCircle;
        }

        return std::make_unique<PlanarDetectorDescriptor>(coeffs, spacing, geometryList);
    }

} // namespace elsa
