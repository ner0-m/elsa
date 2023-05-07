/// Elsa example program: basic 2d X-ray CT simulation and reconstruction

#include "elsa.h"
#include "PhaseContrastProjector.h"

#include <iostream>

using namespace elsa;

void example2d_phase_contrast()
{
    // generate 2d phantom
    IndexVector_t size(2);
    // size << 128, 128;
    size << 512, 512;
    auto phantom = phantoms::modifiedSheppLogan(size);
    auto& volumeDescriptor = phantom.getDataDescriptor();

    // write the phantom out
    io::write(phantom, "2dphantom.pgm");

    // generate circular trajectory
    index_t numAngles{512}, arc{360};
    const auto distance = static_cast<real_t>(size(0));
    auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(
        numAngles, phantom.getDataDescriptor(), arc, distance * 100.0f, distance);

    // dynamic_cast to VolumeDescriptor is legal and will not throw, as Phantoms returns a
    // VolumeDescriptor
    Logger::get("Info")->info("Create PhaseContrastBlobVoxelProjector");
    PhaseContrastBlobVoxelProjector projector(
        dynamic_cast<const VolumeDescriptor&>(volumeDescriptor), *sinoDescriptor);

    // simulate the sinogram
    Logger::get("Info")->info("Calculate sinogram");
    auto sinogram = projector.apply(phantom);

    // write the sinogram out
    Logger::get("Info")->info("Write sinogram");
    io::write(sinogram, "2dsinogram.pgm");

    // solve the reconstruction problem
    CGLS cgSolver(projector, sinogram);

    index_t noIterations{10};
    Logger::get("Info")->info("Solving reconstruction using {} iterations of conjugate gradient",
                              noIterations);
    auto cgReconstruction = cgSolver.solve(noIterations);
    std::cout << cgReconstruction.l2Norm() << "\n";

    // write the reconstruction out
    io::write(cgReconstruction, "2dreconstruction_cg.pgm");
}

int main()
{
    try {
        example2d_phase_contrast();
    } catch (std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << "\n";
    }
}
