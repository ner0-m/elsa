/// Elsa example program: basic temporary 2d X-ray CT simulation and problem setup

#include "elsa.h"

#include <iostream>

using namespace elsa;

void example2d_ctm()
{
    index_t n = 512;
    DataContainer<real_t> image(VolumeDescriptor{{n, n}});
    // TODO image = EDF::read(...);
    const auto& volumeDescriptor = image.getDataDescriptor();

    // generate circular trajectory
    index_t numOfAngles{32}, arc{360};
    real_t sourceFactor = 10000.0f;
    const auto distance = static_cast<real_t>(n);
    auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(
        numOfAngles, volumeDescriptor, arc, distance * sourceFactor, distance);

    SiddonsMethod<real_t> projector(downcast<VolumeDescriptor>(volumeDescriptor), *sinoDescriptor);

    // simulate the sinogram
    auto sinogram = projector.apply(image);

    // TODO what kind of noise did you add to singoram here? e.g. sinogram = singoram + ?

    // setup reconstruction problem
    WLSProblem<real_t> wlsProblem(projector, sinogram);

    // TODO continue here with the remaining steps of the algorithm
}

int main()
{
    try {
        example2d_ctm();
    } catch (std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << "\n";
    }
}
