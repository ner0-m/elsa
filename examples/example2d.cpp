/// Elsa example program: basic 2d X-ray CT simulation and reconstruction

#include "elsa.h"
#include "Trajectories.h"

#include <iostream>

using namespace elsa;

void example2d()
{
    // generate 2d phantom
    IndexVector_t size(2);
    size << 128, 128;
    auto phantom = PhantomGenerator<real_t>::createModifiedSheppLogan(size);
    auto& volumeDescriptor = phantom.getDataDescriptor();

    // write the phantom out
    PGM::write(phantom, "2dphantom.pgm");

    // generate circular trajectory
    auto sinoDescriptor = trajectories::circular(volumeDescriptor);

    // setup operator for 2d X-ray transform
    Logger::get("Info")->info("Simulating sinogram using Siddon's method");

    // dynamic_cast to VolumeDescriptor is legal and will not throw, as PhantomGenerator returns a
    // VolumeDescriptor
    SiddonsMethod projector(dynamic_cast<const VolumeDescriptor&>(volumeDescriptor),
                            sinoDescriptor);

    // simulate the sinogram
    auto sinogram = projector.apply(phantom);

    // write the sinogram out
    PGM::write(sinogram, "2dsinogram.pgm");

    // setup reconstruction problem
    WLSProblem wlsProblem(projector, sinogram);

    // solve the reconstruction problem
    CG cgSolver(wlsProblem);

    index_t noIterations{20};
    Logger::get("Info")->info("Solving reconstruction using {} iterations of conjugate gradient",
                              noIterations);
    auto cgReconstruction = cgSolver.solve(noIterations);

    // write the reconstruction out
    PGM::write(cgReconstruction, "2dreconstruction_cg.pgm");

    LASSOProblem lassoProb(projector, sinogram);

    // solve the reconstruction problem with ISTA
    ISTA istaSolver(lassoProb);

    Logger::get("Info")->info("Solving reconstruction using {} iterations of ISTA", noIterations);
    auto istaReconstruction = istaSolver.solve(noIterations);

    // write the reconstruction out
    PGM::write(istaReconstruction, "2dreconstruction_ista.pgm");

    // solve the reconstruction problem with FISTA
    FISTA fistaSolver(lassoProb);

    Logger::get("Info")->info("Solving reconstruction using {} iterations of FISTA", noIterations);
    auto fistaReconstruction = fistaSolver.solve(noIterations);

    // write the reconstruction out
    PGM::write(fistaReconstruction, "2dreconstruction_fista.pgm");
}

int main()
{
    try {
        example2d();
    } catch (std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << "\n";
    }
}
