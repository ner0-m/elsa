/// elsa example program: basic 2D X-ray CT simulation and reconstruction through shearlet-based
/// ADMM

#include "elsa.h"

#include <iostream>

using namespace elsa;

void example2d_shadmm()
{
    // generate 2d phantom
    index_t n = 128;
    IndexVector_t signalSize(2);
    signalSize << n, n;
    auto phantom = PhantomGenerator<real_t>::createModifiedSheppLogan(signalSize);
    const DataDescriptor& domainDescriptor = phantom.getDataDescriptor();

    // generate circular trajectory
    index_t numOfAngles{360}, arc{360};
    const auto distance = static_cast<real_t>(n);
    auto sinoDescriptor = LimitedAngleTrajectoryGenerator::createTrajectory(
        numOfAngles, std::pair(geometry::Degree(60), geometry::Degree(120)), domainDescriptor, arc,
        distance * 100.0f, distance);

    // consider using the CUDA version if available
    SiddonsMethod<real_t> projector(downcast<VolumeDescriptor>(domainDescriptor), *sinoDescriptor);

    // simulate the sinogram
    auto sinogram = projector.apply(phantom);

    // setup reconstruction problem
    WLSProblem<real_t> wlsProblem(projector, sinogram);

    ShearletTransform<real_t, real_t> shearletTransform(signalSize);
    shearletTransform.computeSpectra();
    index_t layers = shearletTransform.getNumOfLayers();

    /// values specific to the problem statement in T. A. Bubba et al., consider as hyper-parameters
    real_t rho1 = 1.0 / 2;
    real_t rho2 = 1;

    VolumeDescriptor layersPlusOneDescriptor{{n, n, layers + 1}};

    /// AT = (ρ_1*SH^T, ρ_2*I_n^2 ) ∈ R ^ n^2 × (L+1)n^2
    IndexVector_t slicesInBlock(2);
    slicesInBlock << layers, 1;
    std::vector<std::unique_ptr<LinearOperator<real_t>>> opsOfA(0);
    Scaling<real_t> scaling(domainDescriptor, rho2);
    opsOfA.push_back((rho1 * shearletTransform).clone()); // TODO double check
    opsOfA.push_back(scaling.clone());
    BlockLinearOperator<real_t> A(domainDescriptor,
                                  PartitionDescriptor{layersPlusOneDescriptor, slicesInBlock},
                                  opsOfA, BlockLinearOperator<real_t>::BlockType::ROW);

    /// B = diag(−ρ_1*1_Ln^2, −ρ_2*1_n^2) ∈ R ^ (L+1)n^2 × (L+1)n^2
    DataContainer<real_t> factorsOfB(VolumeDescriptor{n, n, layers + 1});
    for (int ind = 0; ind < factorsOfB.getSize(); ++ind) { // TODO double check
        if (ind < (n * n * layers)) {
            factorsOfB[ind] = -1 * rho1;
        } else {
            factorsOfB[ind] = -1 * rho2;
        }
    }
    Scaling<real_t> B(layersPlusOneDescriptor, factorsOfB);

    DataContainer<real_t> c(layersPlusOneDescriptor);
    c = 0;

    Constraint<real_t> constraint(A, B, c);

    // construct shearlet-based l1 regularization
    DataContainer<real_t> wL1NWeights(shearletTransform.getRangeDescriptor());
    wL1NWeights = 0.001f; // TODO address
    WeightedL1Norm<real_t> weightedL1Norm(LinearResidual<real_t>{shearletTransform}, wL1NWeights);
    RegularizationTerm<real_t> wL1NormRegTerm(1, weightedL1Norm);

    // solve the reconstruction problem with ADMM
    SplittingProblem<real_t> splittingProblem(wlsProblem.getDataTerm(), wL1NormRegTerm, constraint);

    ADMM<CG, SoftThresholding, real_t> admm(splittingProblem, true);

    index_t noIterations{5};

    Logger::get("Info")->info("Solving reconstruction using {} iterations of ADMM", noIterations);
    auto solution = admm.solve(noIterations);

    // write the reconstruction out
    EDF::write(solution, "2dreconstruction_sdlx.edf");
}

int main()
{
    try {
        example2d_shadmm();
    } catch (std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << "\n";
    }
}
