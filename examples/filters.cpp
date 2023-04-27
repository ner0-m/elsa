/// Elsa example program: basic 2d X-ray CT simulation and reconstruction using FBP

#include "CartesianIndices.h"
#include "DataContainer.h"
#include "Filter.h"
#include "VolumeDescriptor.h"
#include "elsa.h"
#include "functions/Abs.hpp"

#include <iostream>

using namespace elsa;

void printFilter(std::string_view name, const Filter<float>& filter)
{
    auto shifted = fftShift(filter.getScaleFactors());
    std::cout << name << "=[";
    for (index_t i = 0; i < filter.getDomainDescriptor().getNumberOfCoefficientsPerDimension()[0];
         i++) {
        std::cout << elsa::abs(shifted[i]) << ',';
    }
    std::cout << ']' << std::endl;
}

void filters()
{
    auto coeffs = IndexVector_t{{128, 128}};
    auto desc = VolumeDescriptor{coeffs};

    auto ramlak = makeRamLak(desc);
    io::write(elsa::cwiseAbs(fftShift(ramlak.getScaleFactors())), "filters_RamLak.pgm");

    auto sheppLogan = makeSheppLogan(desc);
    io::write(elsa::cwiseAbs(fftShift(sheppLogan.getScaleFactors())), "filters_SheppLogan.pgm");

    auto cosine = makeCosine(desc);
    io::write(elsa::cwiseAbs(fftShift(cosine.getScaleFactors())), "filters_cosine.pgm");

    auto hann = makeHann(desc);
    io::write(elsa::cwiseAbs(fftShift(hann.getScaleFactors())), "filters_Hann.pgm");

    printFilter("rl", ramlak);
    printFilter("sl", sheppLogan);
    printFilter("cos", cosine);
    printFilter("hann", hann);
}

int main()
{
    try {
        filters();
    } catch (std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << "\n";
    }
}
