/// Elsa example program: basic 2d X-ray CT simulation and reconstruction

#include "elsa.h"
#include "PhantomGenerator.h"
#include "CircleTrajectoryGenerator.h"
#include "BinaryMethod.h"
#include "EDFHandler.h"
#include "GradientDescent.h"
#include "WLSProblem.h"

#include <iostream>

using namespace elsa;

void example2d() {
    // generate 2d phantom of size 128x128
    IndexVector_t size(2); size << 128, 128;
    auto phantom = PhantomGenerator<real_t>::createModifiedSheppLogan(size);

    // write the phantom out
    EDF::write(phantom, "2dphantom.edf");

    // generate circular trajectory with 100 angles over 360 degrees
    auto [geometry, sinoDescriptor] =
    CircleTrajectoryGenerator::createTrajectory(100, phantom.getDataDescriptor(), 360, size(0)*100, size(0));

    // setup operator for 2d X-ray transform
    BinaryMethod projector(phantom.getDataDescriptor(), sinoDescriptor, geometry);

    // simulate the sinogram
    auto sinogram = projector.apply(phantom);

    // write the sinogram out
    EDF::write(sinogram, "2dsinogram.edf");


    // setup reconstruction problem
    WLSProblem problem(projector, sinogram);

    // solve the reconstruction problem
    GradientDescent solver(problem, 1.0/size.prod());
    auto reconstruction = solver.solve(50);

    // write the reconstruction out
    EDF::write(reconstruction, "2dreconstruction.edf");
}


int main() {
    try {
        example2d();
    }
    catch (std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << "\n";
    }
}