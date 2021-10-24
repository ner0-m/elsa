/// Elsa example program: basic 2d X-ray CT simulation and reconstruction

#include "elsa.h"

#include <iostream>

using namespace elsa;

void sparse_sdlx_example2d()
{
    // generate 2d phantom
    IndexVector_t size(2);
    size << 512, 512;
    // TODO do this instead auto image = EDF::read("some_image.edf");
    auto image = PhantomGenerator<real_t>::createModifiedSheppLogan(size);
    const auto& volumeDescriptor = image.getDataDescriptor();

    // write the sinogram out
    EDF::write(image, "image.edf");

    // generate circular trajectory
    // TODO is numAngles the most important thing to look out for in sparse CT?
    index_t numAngles{64}, arc{360};
    const auto distance = static_cast<real_t>(512);
    auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(
        numAngles, volumeDescriptor, arc, distance * 100.0f, distance);

    // setup operator for 2d X-ray transform
    Logger::get("Info")->info("Simulating sinogram using Siddon's method");

    // dynamic_cast to VolumeDescriptor is legal and will not throw, as PhantomGenerator returns a
    // VolumeDescriptor
    SiddonsMethodCUDA<real_t> projector(dynamic_cast<const VolumeDescriptor&>(volumeDescriptor),
                                        *sinoDescriptor);

    // simulate the sinogram
    auto sinogram = projector.apply(image);

    // write the sinogram out
    EDF::write(sinogram, "2dsinogram.edf");

    // TODO ideally use the SDLX method in the following, not CG

    // setup reconstruction problem
    WLSProblem wlsProblem(projector, sinogram);

    // solve the reconstruction problem
    CG cgSolver(wlsProblem);

    index_t noIterations{10};
    Logger::get("Info")->info("Solving reconstruction using {} iterations of conjugate gradient",
                              noIterations);
    auto cgReconstruction = cgSolver.solve(noIterations);

    // write the reconstruction out
    EDF::write(cgReconstruction, "2dreconstruction_cg.edf");
}

int main()
{
    try {
        sparse_sdlx_example2d();
    } catch (std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << "\n";
    }
}
