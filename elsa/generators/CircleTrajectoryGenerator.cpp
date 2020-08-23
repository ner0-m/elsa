#include "CircleTrajectoryGenerator.h"
#include "Logger.h"
#include "VolumeDescriptor.h"
#include "PlanarDetectorDescriptor.h"
#include "CurvedDetectorDescriptor.h"

#include <stdexcept>

namespace elsa
{
    std::unique_ptr<DetectorDescriptor> CircleTrajectoryGenerator::createTrajectory(
        index_t numberOfPoses, const DataDescriptor& volumeDescriptor, index_t arcDegrees,
        real_t sourceToCenter, real_t centerToDetector,
        std::optional<TrajectoryWithCurvedDetector> createWithCurved)
    {
        // pull in geometry namespace, to reduce cluttering
        using namespace geometry;

        // sanity check
        auto dim = volumeDescriptor.getNumberOfDimensions();

        if (dim < 2 || dim > 3)
            throw std::invalid_argument("CircleTrajectoryGenerator: can only handle 2d/3d");

        Logger::get("CircleTrajectoryGenerator")
            ->info("creating {}D trajectory with {} poses in an {} degree arc", dim, numberOfPoses,
                   arcDegrees);

        IndexVector_t coeffs(dim);
        RealVector_t spacing(dim);
        if (dim == 2) {
            coeffs << volumeDescriptor.getNumberOfCoefficientsPerDimension()[0], numberOfPoses;
            spacing << volumeDescriptor.getSpacingPerDimension()[0], 1;
        } else {
            coeffs << volumeDescriptor.getNumberOfCoefficientsPerDimension()[0],
                volumeDescriptor.getNumberOfCoefficientsPerDimension()[1], numberOfPoses;
            spacing << volumeDescriptor.getSpacingPerDimension()[0],
                volumeDescriptor.getSpacingPerDimension()[1], 1;
        }

        std::vector<Geometry> geometryList;

        real_t angleIncrement = static_cast<real_t>(1.0f) * static_cast<real_t>(arcDegrees)
                                / (static_cast<real_t>(numberOfPoses) - 1.0f);
        for (index_t i = 0; i < numberOfPoses; ++i) {
            real_t angle =
                static_cast<real_t>(i) * angleIncrement * pi_t / 180.0f; // convert to radians
            if (dim == 2) {
                Geometry geom(SourceToCenterOfRotation{sourceToCenter},
                              CenterOfRotationToDetector{centerToDetector}, Radian{angle},
                              VolumeData2D{volumeDescriptor.getSpacingPerDimension(),
                                           volumeDescriptor.getLocationOfOrigin()},
                              SinogramData2D{Size2D{coeffs}, Spacing2D{spacing}});
                geometryList.push_back(geom);
            } else {
                Geometry geom(SourceToCenterOfRotation{sourceToCenter},
                              CenterOfRotationToDetector{centerToDetector},
                              VolumeData3D{volumeDescriptor.getSpacingPerDimension(),
                                           volumeDescriptor.getLocationOfOrigin()},
                              SinogramData3D{Size3D{coeffs}, Spacing3D{spacing}},
                              RotationAngles3D{Radian{angle}, Radian{0}, Radian{0}});
                geometryList.push_back(geom);
            }
        }

        if (createWithCurved)
            // TODO: this scaling is arbitary so far. We have to see if this is the format needed,
            // or at least such that we can translate it into this format
            return std::make_unique<CurvedDetectorDescriptor>(coeffs, spacing, geometryList, 1.0);
        else
            return std::make_unique<PlanarDetectorDescriptor>(coeffs, spacing, geometryList);
    }

} // namespace elsa
