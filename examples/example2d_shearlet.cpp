#include "elsa.h"

#include <iostream>

using namespace elsa;

void shearlet_example()
{
    // generate 2d phantom
    IndexVector_t size(2);
    size << 256, 256;
    auto phantom = PhantomGenerator<real_t>::createModifiedSheppLogan(size);

    ShearletTransform<real_t, real_t> shearletTransform(size);

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
