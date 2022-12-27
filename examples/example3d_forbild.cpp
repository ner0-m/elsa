/// Elsa example program: basic 3d X-ray CT simulation and reconstruction using CUDA projectors

#include "elsa.h"
#include "Logger.h"
#include <Error.h>
#include <CylinderFree.h>
#include <Box.h>

#include <iostream>
#include <filesystem>

using namespace elsa;

void write(elsa::DataContainer<double>& phantom, std::string prae, int dimension)
{
    if (!std::filesystem::exists(std::filesystem::path{prae})) {
        int status = mkdir(prae.data(), 777);
        if (status != 0) {
            Logger::get("Examplle 3d forbild")->error("Can not create directory {}", prae);
            return;
        }
    }

    int max = 100;
    for (int i = 0; i < max; i++) {

        std::string num = std::to_string(i);
        // write the slice
        std::string name;
        if (i < 10) {
            name = std::string("0") + num + ".pgm";
        } else {
            name = num + ".pgm";
        }
        int sli = (i * dimension) / max;
        io::write(phantom.slice(sli), prae + "/" + name);
        Logger::get("Ellipsoid")->info("Export to {} slice {}", prae + "/" + name, sli);
    }
}

void example3d()
{

    int dimension = 200;

    // generate 3d phantom
    IndexVector_t size(3);
    size << dimension, dimension, dimension;

    auto phantom = phantoms::forbild_head<double>(size);
    write(phantom, std::string{"head"}, dimension);
    phantom = 0;

    phantom = phantoms::forbild_thorax<double>(size);
    write(phantom, std::string{"thorax"}, dimension);
    phantom = 0;

    phantom = phantoms::forbild_abdomen<double>(size);
    write(phantom, std::string{"abdomen"}, dimension);
}

int main()
{
    try {
        example3d();
    } catch (elsa::Error& e) {
        std::cerr << "An exception occurred: " << e << "\n";
    }
}
