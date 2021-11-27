/// Elsa example program: basic 2d X-ray CT simulation and reconstruction

#include "elsa.h"

#include "SHADMM.h"
#include "NoiseGenerators.h"

#include <iostream>

using namespace elsa;

void example2d_shadmm()
{
    // generate 2d phantom
    index_t n = 128;
    IndexVector_t signalSize(2);
    signalSize << n, n;
    auto phantom = PhantomGenerator<real_t>::createModifiedSheppLogan(signalSize);

    // generate circular trajectory
    index_t numOfAngles{360}, arc{360};
    const auto distance = static_cast<real_t>(n);
    auto sinoDescriptor = LimitedAngleTrajectoryGenerator::createTrajectory(
        numOfAngles, std::pair(geometry::Degree(60), geometry::Degree(120)),
        phantom.getDataDescriptor(), arc, distance * 100.0f, distance);

    SiddonsMethodCUDA<real_t> projector(VolumeDescriptor{{n, n}}, *sinoDescriptor);

    // simulate the sinogram
    auto sinogram = projector.apply(phantom);

    // setup reconstruction problem
    WLSProblem<real_t> wlsProblem(projector, sinogram);

    ShearletTransform<real_t, real_t> shearletTransform(n, n);
    shearletTransform.computeSpectra(); // TODO remove to further simplify
    index_t layers = shearletTransform.getNumOfLayers();

    DataContainer<real_t> wL1NWeights(shearletTransform.getRangeDescriptor());
    wL1NWeights = 0.001f;
    WeightedL1Norm<real_t> weightedL1Norm(LinearResidual<real_t>{shearletTransform}, wL1NWeights);
    RegularizationTerm<real_t> wL1NormRegTerm(1, weightedL1Norm);

    // solve the reconstruction problem with ADMM
    SplittingProblem<real_t> splittingProblem(wlsProblem.getDataTerm(), wL1NormRegTerm,
                                              VolumeDescriptor{{n, n, layers + 1}});

    ADMM<CG, SoftThresholding, real_t> shadmm(splittingProblem, true);

    index_t noIterations{5};

    Logger::get("Info")->info("Solving reconstruction using {} iterations of ADMM", noIterations);
    auto sol = shadmm.solve(5);

    // write the reconstruction out
    EDF::write(sol, "2dreconstruction_sdlx.edf");
}

int main()
{
    try {
        example2d_shadmm();
    } catch (std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << "\n";
    }
}
