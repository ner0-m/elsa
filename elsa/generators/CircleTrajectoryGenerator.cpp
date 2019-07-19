#include "CircleTrajectoryGenerator.h"
#include "Logger.h"

#include <stdexcept>

namespace elsa
{
    std::pair<std::vector<Geometry>, DataDescriptor> CircleTrajectoryGenerator::createTrajectory(index_t numberOfPoses,
                                                                      const DataDescriptor& volumeDescriptor,
                                                                      index_t arcDegrees, real_t sourceToCenter, real_t centerToDetector)
    {
        // sanity check
        auto dim = volumeDescriptor.getNumberOfDimensions();

        if (dim < 2 || dim > 3)
            throw std::invalid_argument("CircleTrajectoryGenerator: can only handle 2d/3d");

        Logger::get("CircleTrajectoryGenerator")->info("creating {}D trajectory with {} poses in an {} degree arc", dim, numberOfPoses, arcDegrees);

        IndexVector_t coeffs(dim);
        RealVector_t spacing(dim);
        if (dim == 2) {
            coeffs  << volumeDescriptor.getNumberOfCoefficientsPerDimension()[0], numberOfPoses;
            spacing << volumeDescriptor.getSpacingPerDimension()[0], 1;
        }
        else {
            coeffs  << volumeDescriptor.getNumberOfCoefficientsPerDimension()[0],
                       volumeDescriptor.getNumberOfCoefficientsPerDimension()[1], numberOfPoses;
            spacing << volumeDescriptor.getSpacingPerDimension()[0],
                       volumeDescriptor.getSpacingPerDimension()[1], 1;
        }
        DataDescriptor sinoDescriptor(coeffs, spacing);

        std::vector<Geometry> geometryList;

        real_t angleIncrement = static_cast<real_t>(1.0) * arcDegrees / (numberOfPoses-1);
        for (index_t i = 0; i < numberOfPoses; ++i) {
            real_t angle = i * angleIncrement * pi / 180.0; // convert to radians
            if (dim == 2) {
                Geometry geom(sourceToCenter, centerToDetector, angle, volumeDescriptor, sinoDescriptor);
                geometryList.push_back(geom);
            }
            else {
                Geometry geom(sourceToCenter, centerToDetector, volumeDescriptor, sinoDescriptor, angle);
                geometryList.push_back(geom);
            }
        }

        return std::make_pair(geometryList, sinoDescriptor);
    }

} // namespace elsa
