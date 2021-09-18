#include "elsa.h"

#include <iostream>

using namespace elsa;

void shearlet_example()
{
    // generate 2d phantom
    IndexVector_t size(2);
    size << 128, 128;
    auto phantom = PhantomGenerator<real_t>::createModifiedSheppLogan(size);
    EDF::write(phantom, "128phantom.edf");

    ShearletTransform<real_t> shearletTransform(size[0], size[1], 5);

    Logger::get("Info")->info("Applying shearlet transform");
    DataContainer<real_t> shearletCoefficients = shearletTransform.apply(phantom);

    Logger::get("Info")->info("Applying inverse shearlet transform");
    DataContainer<real_t> reconstruction = shearletTransform.applyAdjoint(shearletCoefficients);

    // write the reconstruction out
    EDF::write(reconstruction, "2dreconstruction_shearlet.edf");
}

int main()
{
    try {
        shearlet_example();
    } catch (std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << "\n";
    }
}