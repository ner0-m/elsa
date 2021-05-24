#include "elsa.h"

#include <iostream>

using namespace elsa;

void shearlet_example()
{
    // TODO get an example image, from where? use existing? Shepp-Logan?
    DataContainer<real_t> image(VolumeDescriptor{{511, 511}});
    const auto& imageDescriptor = image.getDataDescriptor();

    ConeAdaptedDiscreteShearletTransform<real_t> shearletTransform(
        imageDescriptor.getNumberOfCoefficientsPerDimension()[0],
        imageDescriptor.getNumberOfCoefficientsPerDimension()[1]);

    Logger::get("Info")->info("Applying shearlet transform");
    DataContainer<real_t> shearletCoefficients = shearletTransform.apply(image);

    Logger::get("Info")->info("Applying inverse shearlet transform");
    // TODO note that the initial output might not be entirely reals, might be complex
    DataContainer<real_t> reconstruction = shearletTransform.applyAdjoint(shearletCoefficients);

    // TODO display image
    // TODO EDF write?
}

int main()
{
    try {
        shearlet_example();
    } catch (std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << "\n";
    }
}
