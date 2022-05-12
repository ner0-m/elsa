/// Elsa example program: basic 2d X-ray CT simulation and reconstruction

#include "elsa.h"
#include "Tasks.hpp"

#include <iostream>

using namespace elsa;

/// train on x and y, test on image
DataContainer<float> runTheSDLXTask(const std::vector<DataContainer<float>>& x,
                                    const std::vector<DataContainer<float>>& y,
                                    DataContainer<float> image)
{
    float distance =
        static_cast<float>(image.getDataDescriptor().getNumberOfCoefficientsPerDimension()[0]);

    std::unique_ptr<DetectorDescriptor> trajectory =
        task::ConstructTrajectory::circle.setNumberOfPoses(360)
            .setArcDegrees(360)
            .setSourceToCenter(distance * 100)
            .setCenterToDetector(distance)
            .build(image.getDataDescriptor());

    DataContainer<float> visibleCoeffs =
        task::InpaintMissingSingularities::reconstructVisibleCoeffs(image, std::move(trajectory));

    ml::Model trainedModel = task::InpaintMissingSingularities::inpaintInvisibleCoeffs(x, y);

    auto result = task::InpaintMissingSingularities::combineVisCoeffsWithInpaintedInvisCoeffs(
        visibleCoeffs, trainedModel.predict(image));

    return result;

    // or just do
    // return task::InpaintMissingSingularities::run(x, y, std::move(trajectory), {image})[0];
}

int main()
{
    try {
        DataContainer<float> image(VolumeDescriptor{28, 28, 1});
        image = 0;

        const std::vector<DataContainer<float>>& x = {image};
        const std::vector<DataContainer<float>>& y = {image};

        runTheSDLXTask(x, y, image);
    } catch (std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << "\n";
    }
}
