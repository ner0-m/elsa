/// Elsa example program: basic 2d X-ray CT simulation and reconstruction

#include "elsa.h"

#include <iostream>

using namespace elsa;

void example2d()
{
    // generate 2d phantom
    IndexVector_t size(2); size << 128, 128;
    auto phantom = PhantomGenerator<real_t>::createModifiedSheppLogan(size);

    // write the phantom out
    EDF::write(phantom, "2dphantom.edf");

    // generate circular trajectory
    index_t noAngles{100}, arc{360};
    auto [geometry, sinoDescriptor] = CircleTrajectoryGenerator::createTrajectory(noAngles, phantom.getDataDescriptor(),
            arc, size(0)*100, size(0));

    // setup operator for 2d X-ray transform
    Logger::get("Info")->info("Simulating sinogram using Siddon's method");
    SiddonsMethod projector(phantom.getDataDescriptor(), sinoDescriptor, geometry);

    // simulate the sinogram
    auto sinogram = projector.apply(phantom);

    // write the sinogram out
    EDF::write(sinogram, "2dsinogram.edf");


    // setup reconstruction problem
    WLSProblem problem(projector, sinogram);

    // solve the reconstruction problem
    GradientDescent solver(problem, 1.0/size.prod());

    index_t noIterations{50};
    Logger::get("Info")->info("Solving reconstruction using {} iterations of gradient descent", noIterations);
    auto reconstruction = solver.solve(noIterations);

    // write the reconstruction out
    EDF::write(reconstruction, "2dreconstruction.edf");
}


int main()
{
    try {
        example2d();
    }
    catch (std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << "\n";
    }
}
