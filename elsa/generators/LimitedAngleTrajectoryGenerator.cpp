#include "LimitedAngleTrajectoryGenerator.h"
#include "VolumeDescriptor.h"
#include "PlanarDetectorDescriptor.h"
#include "Logger.h"

#include <stdexcept>

namespace elsa
{
    std::unique_ptr<DetectorDescriptor> LimitedAngleTrajectoryGenerator::createTrajectory(
        index_t numberOfPoses,
        std::pair<elsa::geometry::Degree, elsa::geometry::Degree> missingWedgeAngles,
        const DataDescriptor& volumeDescriptor, index_t arcDegrees, real_t sourceToCenter,
        real_t centerToDetector, bool mirrored)
    {
        // pull in geometry namespace, to reduce cluttering
        using namespace geometry;

        // sanity check
        const auto dim = volumeDescriptor.getNumberOfDimensions();
        if (dim != 2) {
            throw InvalidArgumentError("LimitedAngleTrajectoryGenerator: can only handle 2D");
        }

        Logger::get("LimitedAngleTrajectoryGenerator")
            ->info("creating 2D trajectory with {} poses in an {} degree arc", numberOfPoses,
                   arcDegrees);

        const auto [coeffs, spacing] =
            calculateSizeAndSpacingPerGeometry(volumeDescriptor, numberOfPoses);

        // create vector and reserve the necessary size, minor optimization such that no new
        // allocations are necessary in the loop
        std::vector<Geometry> geometryList;
        geometryList.reserve(static_cast<std::size_t>(numberOfPoses));

        real_t wedgeArc = mirrored ? 2 * (missingWedgeAngles.second - missingWedgeAngles.first)
                                   : missingWedgeAngles.second - missingWedgeAngles.first;

        const real_t angleIncrement = (static_cast<real_t>(arcDegrees) - wedgeArc)
                                      / (static_cast<real_t>(numberOfPoses) - 1.0f);

        for (index_t i = 0;; ++i) {
            Radian angle = Degree{static_cast<real_t>(i) * angleIncrement};

            if (notInMissingWedge(angle, missingWedgeAngles, mirrored)) {
                // use emplace_back, then no copy is created
                geometryList.emplace_back(SourceToCenterOfRotation{sourceToCenter},
                                          CenterOfRotationToDetector{centerToDetector},
                                          Radian{angle},
                                          VolumeData2D{volumeDescriptor.getSpacingPerDimension(),
                                                       volumeDescriptor.getLocationOfOrigin()},
                                          SinogramData2D{Size2D{coeffs}, Spacing2D{spacing}});
            }

            if (angle.to_degree() > static_cast<real_t>(arcDegrees)) {
                break;
            }
        }

        return std::make_unique<PlanarDetectorDescriptor>(coeffs, spacing, geometryList);
    }

    bool LimitedAngleTrajectoryGenerator::notInMissingWedge(
        elsa::geometry::Radian angle,
        std::pair<elsa::geometry::Degree, elsa::geometry::Degree> missingWedgeAngles, bool mirrored)
    {
        if (!mirrored) {
            return !(angle.to_degree() >= missingWedgeAngles.first
                     && angle.to_degree() <= missingWedgeAngles.second);
        } else {
            return !((angle.to_degree() >= missingWedgeAngles.first
                      && angle.to_degree() <= missingWedgeAngles.second)
                     || (angle.to_degree() >= (missingWedgeAngles.first + 180)
                         && angle.to_degree() <= (missingWedgeAngles.second + 180)));
        }
    }
} // namespace elsa
