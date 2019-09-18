/// Elsa example program: basic 3d X-ray CT simulation and reconstruction using CUDA projectors

#include "elsa.h"
#include "PhantomGenerator.h"
#include "CircleTrajectoryGenerator.h"
#include "JosephsMethodCUDA.h"
#include "EDFHandler.h"
#include "GradientDescent.h"
#include "WLSProblem.h"
#include "Logger.h"

#include <iostream>

using namespace elsa;

void example2d()
{
    // generate 3d phantom
    IndexVector_t size(3); size << 128, 128, 128;
    auto phantom = PhantomGenerator<real_t>::createModifiedSheppLogan(size);

    // write the phantom out
    EDF::write(phantom, "3dphantom.edf");

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
    EDF::write(sinogram, "3dsinogram.edf");


    // setup reconstruction problem
    WLSProblem problem(projector, sinogram);

    // solve the reconstruction problem
    GradientDescent solver(problem, 1.0/size.prod());

    index_t noIterations{50};
    Logger::get("Info")->info("Solving reconstruction using {} iterations of gradient descent", noIterations);
    auto reconstruction = solver.solve(noIterations);

    // write the reconstruction out
    EDF::write(reconstruction, "3dreconstruction.edf");
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