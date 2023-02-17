/// Elsa example program: basic 3d X-ray CT simulation and reconstruction using CUDA projectors

#include "elsa.h"

#include <iostream>

using namespace elsa;

void example3d()
{
    // generate 3d phantom
    IndexVector_t size(3);
    size << 128, 128, 128;
    auto phantom = phantoms::modifiedSheppLogan<real_t>(size);
    auto& volumeDescriptor = phantom.getDataDescriptor();

    // write the phantom out
    EDF::write(phantom, "3dphantom.edf");

    // generate circular trajectory
    index_t numAngles{180}, arc{360};
    real_t distance = static_cast<real_t>(size(0));
    auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(
        numAngles, phantom.getDataDescriptor(), arc, distance * 100.0f, distance);

    // setup operator for 2d X-ray transform
    Logger::get("Info")->info("Simulating sinogram using Siddon's method");

    // dynamic_cast to VolumeDescriptor is legal and will not throw, as Phantoms returns a
    // VolumeDescriptor
    JosephsMethodCUDA projector(downcast<VolumeDescriptor>(volumeDescriptor), *sinoDescriptor);

    // simulate the sinogram
    auto sinogram = projector.apply(phantom);

    // write the sinogram out
    EDF::write(sinogram, "3dsinogram.edf");

    // solve the reconstruction problem
    CGLS solver(projector, sinogram);

    index_t noIterations{20};
    Logger::get("Info")->info("Solving reconstruction using {} iterations of conjugate gradient",
                              noIterations);
    auto reconstruction = solver.solve(noIterations);

    // write the reconstruction out
    EDF::write(reconstruction, "3dreconstruction.edf");
}

int main()
{
    try {
        example3d();
    } catch (std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << "\n";
    }
}
