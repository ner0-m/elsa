/// Elsa example program: basic 2d X-ray CT simulation and reconstruction

#include "elsa.h"
#include "PhantomGenerator.h"
#include "CircleTrajectoryGenerator.h"
#include "BinaryMethod.h"
#include "EDFHandler.h"
#include "GradientDescent.h"
#include "WLSProblem.h"
#include "Logger.h"

#include <iostream>

using namespace elsa;

void example2d()
{
    // generate 2d phantom
    IndexVector_t size(2); size << 128, 128;
    Logger::get("Info")->info("Creating 2d phantom of size {}x{}", size(0), size(1));
    auto phantom = PhantomGenerator<real_t>::createModifiedSheppLogan(size);

    // write the phantom out
    std::string filePhantom{"2dphantom.edf"};
    Logger::get("Info")->info("Writing out phantom to {}", filePhantom);
    EDF::write(phantom, filePhantom);

    // generate circular trajectory
    index_t noAngles{100}, arc{360};
    Logger::get("Info")->info("Creating circular trajectory with {} poses over an arc of {} degrees", noAngles, arc);
    auto [geometry, sinoDescriptor] = CircleTrajectoryGenerator::createTrajectory(noAngles, phantom.getDataDescriptor(),
            arc, size(0)*100, size(0));

    // setup operator for 2d X-ray transform
    Logger::get("Info")->info("Simulating sinogram using binary method");
    BinaryMethod projector(phantom.getDataDescriptor(), sinoDescriptor, geometry);

    // simulate the sinogram
    auto sinogram = projector.apply(phantom);

    // write the sinogram out
    std::string fileSinogram("2dsinogram.edf");
    Logger::get("Info")->info("Writing out phantom to {}", fileSinogram);
    EDF::write(sinogram, fileSinogram);


    // setup reconstruction problem
    WLSProblem problem(projector, sinogram);

    // solve the reconstruction problem
    GradientDescent solver(problem, 1.0/size.prod());

    index_t noIterations{50};
    Logger::get("Info")->info("Solving reconstruction using {} iterations of gradient descent", noIterations);
    auto reconstruction = solver.solve(noIterations);

    // write the reconstruction out
    std::string fileReconstruction{"2dreconstruction.edf"};
    Logger::get("Info")->info("Writing out reconstruction to {}", fileReconstruction);
    EDF::write(reconstruction, fileReconstruction);
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