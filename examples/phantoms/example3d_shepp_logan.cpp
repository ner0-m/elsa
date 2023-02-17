/// Elsa example program: basic 3d X-ray CT simulation and reconstruction using CUDA projectors

#include "elsa.h"
#include "Logger.h"
#include <iostream>
#include <Error.h>

using namespace elsa;

void example3d()
{
    int slices = 10;
    int dimension = 20;

    // generate 3d phantom
    IndexVector_t size(3);
    size << dimension, dimension, dimension;

    auto phantom = phantoms::modifiedSheppLogan<real_t>(size);

    for (int i = 0; i * slices < dimension; i++) {

        std::string num = std::to_string(i);
        // write the slice
        std::string name;
        if (i < 10) {
            name = std::string("0") + num + "_shepp_logan_slice.pgm";
        } else {
            name = num + "_shepp_logan_slice.pgm";
        }
        io::write(phantom.slice(i), name);
        Logger::get("Ellipsoid")->info("Export to {}", name);
    }
}

int main()
{
    try {
        example3d();
    } catch (elsa::Error& e) {
        std::cerr << "An exception occurred: " << e << "\n";
    }
}
