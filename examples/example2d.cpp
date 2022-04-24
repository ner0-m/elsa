/// Elsa example program: basic 2d X-ray CT simulation and reconstruction

#include "elsa.h"
#include "LutProjector.h"

#include <iostream>

using namespace elsa;

void example2d()
{
    // generate 2d phantom
    IndexVector_t size(2);
    // size << 128, 128;
    size << 512, 512;
    auto phantom = PhantomGenerator<real_t>::createModifiedSheppLogan(size);
    auto& volumeDescriptor = phantom.getDataDescriptor();

    // write the phantom out
    io::write(phantom, "2dphantom.pgm");

    // generate circular trajectory
    index_t numAngles{512}, arc{360};
    const auto distance = static_cast<real_t>(size(0));
    auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(
        numAngles, phantom.getDataDescriptor(), arc, distance * 100.0f, distance);

    // dynamic_cast to VolumeDescriptor is legal and will not throw, as PhantomGenerator returns a
    // VolumeDescriptor
    Logger::get("Info")->info("Create BlobProjector");
    BlobProjector projector(dynamic_cast<const VolumeDescriptor&>(volumeDescriptor),
                            *sinoDescriptor);

    // simulate the sinogram
    Logger::get("Info")->info("Calculate sinogram");
    auto sinogram = projector.apply(phantom);

    // write the sinogram out
    Logger::get("Info")->info("Write sinogram");
    io::write(sinogram, "2dsinogram.pgm");

    // setup reconstruction problem
    WLSProblem wlsProblem(projector, sinogram);

    // solve the reconstruction problem
    CG cgSolver(wlsProblem);

    index_t noIterations{10};
    Logger::get("Info")->info("Solving reconstruction using {} iterations of conjugate gradient",
                              noIterations);
    auto cgReconstruction = cgSolver.solve(noIterations);
    std::cout << cgReconstruction.l2Norm() << "\n";

    // write the reconstruction out
    io::write(cgReconstruction, "2dreconstruction_cg.pgm");

    // LASSOProblem lassoProb(projector, sinogram);
    //
    // // solve the reconstruction problem with ISTA
    // ISTA istaSolver(lassoProb);
    //
    // Logger::get("Info")->info("Solving reconstruction using {} iterations of ISTA",
    // noIterations); auto istaReconstruction = istaSolver.solve(noIterations);
    //
    // // write the reconstruction out
    // EDF::write(istaReconstruction, "2dreconstruction_ista.edf");
    //
    // // solve the reconstruction problem with FISTA
    // FISTA fistaSolver(lassoProb);
    //
    // Logger::get("Info")->info("Solving reconstruction using {} iterations of FISTA",
    // noIterations); auto fistaReconstruction = fistaSolver.solve(noIterations);
    //
    // // write the reconstruction out
    // EDF::write(fistaReconstruction, "2dreconstruction_fista.edf");
}

int main()
{
    try {
        example2d();
    } catch (std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << "\n";
    }
}
