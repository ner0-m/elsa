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
        const auto [coeffs, spacing] = [&] {
            IndexVector_t coeffs(dim);
            RealVector_t spacing(dim);

            // Scale coeffsPerDim by sqrt(2), this reduces undersampling of the corners, as the
            // detector is larger than the volume. Cast back and forthe to reduce warnings...
            // This has to be a RealVector_t, most likely that the cast happens, anyway we get
            // errors down the line see #86 in Gitlab
            const RealVector_t coeffsPerDim =
                volumeDescriptor.getNumberOfCoefficientsPerDimension().template cast<real_t>();
            const real_t sqrt2 = std::sqrt(2.f);
            const auto coeffsPerDimSclaed = (coeffsPerDim * sqrt2).template cast<index_t>();

            const auto spacingPerDim = volumeDescriptor.getSpacingPerDimension();

            coeffs.head(dim - 1) = coeffsPerDimSclaed.head(dim - 1);
            coeffs[dim - 1] = numberOfPoses; // TODO: with eigen 3.4: `coeffs(Eigen::last) = 1`

            spacing.head(dim - 1) = spacingPerDim.head(dim - 1);
            spacing[dim - 1] = 1; // TODO: same as coeffs

            // Return a pair, then split it using structured bindings
            return std::pair{coeffs, spacing};
        }();

        // Create vector and reserve the necessary size, minor optimization such that no new
        // allocations are necessary in the loop
        std::vector<Geometry> geometryList;
        geometryList.reserve(static_cast<std::size_t>(numberOfPoses));

        const real_t angleIncrement =
            static_cast<real_t>(arcDegrees) / (static_cast<real_t>(numberOfPoses) - 1.0f);
        for (index_t i = 0; i < numberOfPoses; ++i) {
            const Radian angle = Degree{static_cast<real_t>(i) * angleIncrement};
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
