/// Elsa example program: basic 2d X-ray CT simulation and reconstruction through ADMM

#include "elsa.h"

#include <iostream>

using namespace elsa;

void example2d_admm()
{
    // generate 2d phantom
    IndexVector_t size(2);
    size << 128, 128;
    auto phantom = PhantomGenerator<real_t>::createModifiedSheppLogan(size);
    const auto& volumeDescriptor = phantom.getDataDescriptor();

    // write the phantom out
    EDF::write(phantom, "2dphantom.edf");

    // generate circular trajectory
    index_t numOfAngles{180}, arc{360};
    const auto distance = static_cast<real_t>(size(0));
    auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(
        numOfAngles, volumeDescriptor, arc, distance * 100.0f, distance);

    // setup operator for 2d X-ray transform
    Logger::get("Info")->info("Simulating sinogram using Siddon's method");

    // dynamic_cast to VolumeDescriptor is legal and will not throw, as PhantomGenerator returns a
    // VolumeDescriptor
    SiddonsMethod projector(downcast<VolumeDescriptor>(volumeDescriptor), *sinoDescriptor);

    // simulate the sinogram
    auto sinogram = projector.apply(phantom);

    // write the sinogram out
    EDF::write(sinogram, "2dsinogram.edf");

    // setup reconstruction problem
    WLSProblem<real_t> wlsProblem(projector, sinogram);

    L1Norm<real_t> regFunc(projector.getDomainDescriptor());
    RegularizationTerm<real_t> regTerm(0.5f, regFunc);

    // solve the reconstruction problem with ADMM
    SplittingProblem<real_t> splittingProblem(wlsProblem.getDataTerm(), regTerm);

    ADMM<CG, SoftThresholding> admm(splittingProblem);

    index_t noIterations{5};

    Logger::get("Info")->info("Solving reconstruction using {} iterations of ADMM", noIterations);
    auto admmReconstruction = admm.solve(noIterations);

    // write the reconstruction out
    EDF::write(admmReconstruction, "2dreconstruction_admm.edf");
}

int main()
{
    try {
        example2d_admm();
    } catch (std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << "\n";
    }
}
