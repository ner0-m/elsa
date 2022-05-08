/// Elsa example program: basic 2d X-ray CT simulation and reconstruction

#include "elsa.h"
#include "LutProjector.h"
#include "Tasks.hpp"

#include <iostream>

using namespace elsa;

/// train on x and y, test on image
DataContainer<float> runTheSDLXTask(const std::vector<DataContainer<float>>& x,
                                    const std::vector<DataContainer<float>>& y,
                                    DataContainer<float> image)
{
    auto trajectory = task::ConstructTrajectory::circle
                          ->setDistance(static_cast<float>(
                              image.getDataDescriptor().getNumberOfCoefficientsPerDimension()[0]))
                          ->setNumberOfPoses(360)
                          ->setArcDegrees(360)
                          ->setDataDescriptor(image.getDataDescriptor())
                          ->build();

    auto visibleCoeffs =
        task::InpaintMissingSingularities::reconstructVisibleCoeffs(x, trajectory);

    auto trainedModel = task::InpaintMissingSingularities::inpaintInvisibleCoeffs(x, y);

    auto result = task::InpaintMissingSingularities::combineVisCoeffsWithInpaintedInvisCoeffs(
        visibleCoeffs, trainedModel.predict(image));

    return result;
}

int main()
{
    try {
        const std::vector<DataContainer<float>>& x = {};
        const std::vector<DataContainer<float>>& y = {};
        DataContainer<float> image(VolumeDescriptor{1});

        runTheSDLXTask(x, y, image);
    } catch (std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << "\n";
    }
}
