#include "LimitedAngleTrajectoryGenerator.h"
#include "Logger.h"
#include "VolumeDescriptor.h"
#include "PlanarDetectorDescriptor.h"

#include <stdexcept>

namespace elsa
{
    std::unique_ptr<DetectorDescriptor> LimitedAngleTrajectoryGenerator::createTrajectory(
        index_t numberOfPoses, RealVector_t missingWedgeAngles,
        const DataDescriptor& volumeDescriptor, index_t arcDegrees, real_t sourceToCenter,
        real_t centerToDetector)
    {
        // pull in geometry namespace, to reduce cluttering
        using namespace geometry;

        // sanity check
        const auto dim = volumeDescriptor.getNumberOfDimensions();
        if (dim != 2) {
            throw InvalidArgumentError("LimitedAngleTrajectoryGenerator: can only handle 2D");
        }

        printf("the size of the missingWedgeIndices is %ld\n", missingWedgeAngles.size());
        if (missingWedgeAngles.size() != 2) {
            throw InvalidArgumentError("LimitedAngleTrajectoryGenerator: provide only two indices "
                                       "for specifying the missing wedge");
        }

        Logger::get("LimitedAngleTrajectoryGenerator")
            ->info("creating 2D trajectory with {} poses in an {} degree arc", numberOfPoses,
                   arcDegrees);

        // Calculate size and spacing for each geometry pose using a IIFE
        const auto [coeffs, spacing] = [&] {
            IndexVector_t coeffs(dim);
            RealVector_t spacing(dim);

            const RealVector_t coeffsPerDim =
                volumeDescriptor.getNumberOfCoefficientsPerDimension().template cast<real_t>();
            const real_t sqrt2 = std::sqrt(2.f);
            const auto coeffsPerDimScaled = (coeffsPerDim * sqrt2).template cast<index_t>();

            const auto spacingPerDim = volumeDescriptor.getSpacingPerDimension();

            coeffs.head(dim - 1) = coeffsPerDimScaled.head(dim - 1);
            coeffs[dim - 1] = numberOfPoses; // TODO: with eigen 3.4: `coeffs(Eigen::last) = 1`

            spacing.head(dim - 1) = spacingPerDim.head(dim - 1);
            spacing[dim - 1] = 1; // TODO: same as coeffs

            // return a pair, then split it using structured bindings
            return std::pair{coeffs, spacing};
        }();

        // create vector and reserve the necessary size, minor optimization such that no new
        // allocations are necessary in the loop
        std::vector<Geometry> geometryList;
        geometryList.reserve(static_cast<std::size_t>(numberOfPoses));

        const real_t angleIncrement =
            static_cast<real_t>(arcDegrees) / (static_cast<real_t>(numberOfPoses) - 1.0f);
        for (index_t i = 0; i < numberOfPoses; ++i) {
            const Radian angle = Degree{static_cast<real_t>(i) * angleIncrement};
            if ((angle.to_degree() >= missingWedgeAngles[0]
                 && angle.to_degree() <= missingWedgeAngles[1])
                || (angle.to_degree() >= (missingWedgeAngles[0] + 180)
                    && angle.to_degree() <= (missingWedgeAngles[1] + 180))) {
                // express here the missing wedge

                // TODO just add a continue; or a custom Geometry object (different arguments than
                //  below) that represents this missing part?
            } else {
                // use emplace_back, then no copy is created
                geometryList.emplace_back(SourceToCenterOfRotation{sourceToCenter},
                                          CenterOfRotationToDetector{centerToDetector},
                                          Radian{angle},
                                          VolumeData2D{volumeDescriptor.getSpacingPerDimension(),
                                                       volumeDescriptor.getLocationOfOrigin()},
                                          SinogramData2D{Size2D{coeffs}, Spacing2D{spacing}});
            }
        }

        return std::make_unique<PlanarDetectorDescriptor>(coeffs, spacing, geometryList);
    }
} // namespace elsa
