/// Elsa example program: basic 2d X-ray CT simulation and reconstruction

#include "elsa.h"

#include "SHADMM.h"
#include "NoiseGenerators.h"

#include <iostream>

using namespace elsa;

void example2d_shadmm()
{
    // generate 2d phantom
    index_t n = 512;
    IndexVector_t signalSize(2);
    signalSize << n, n;
    auto phantom = PhantomGenerator<real_t>::createModifiedSheppLogan(signalSize);

    // random numbers
    real_t rho1 = 1.0 / 2;
    real_t rho2 = 1; /// LtI fixes this to 1

    /// AT = (ρ_1*SH^T, ρ_2*I_n^2 ) ∈ R ^ n^2 × (L+1)n^2
    ShearletTransform<real_t> shearletTransform(n, n);
    shearletTransform.computeSpectra();
    index_t layers = shearletTransform.getNumOfLayers();

    IndexVector_t slicesInBlock(2);
    slicesInBlock << layers, 1;

    std::vector<std::unique_ptr<LinearOperator<real_t>>> opsOfA(0);
    Scaling<real_t> scaling(VolumeDescriptor{{n, n}}, rho2);
    opsOfA.push_back(shearletTransform.clone()); // TODO mult. with rho1 later on
    opsOfA.push_back(scaling.clone());
    BlockLinearOperator<real_t> A{
        VolumeDescriptor{{n, n}},
        PartitionDescriptor{VolumeDescriptor{{n, n, layers + 1}}, slicesInBlock}, opsOfA,
        BlockLinearOperator<real_t>::BlockType::ROW};

    /// B = diag(−ρ_1*1_Ln^2 , −ρ_2*1_n^2) ∈ R ^ (L+1)n^2 × (L+1)n^2
    DataContainer<real_t> factorsOfB(VolumeDescriptor{n, n, layers + 1});
    for (int ind = 0; ind < factorsOfB.getSize(); ++ind) {
        if (ind < (n * n * layers)) {
            factorsOfB[ind] = -1 * rho1;
        } else {
            factorsOfB[ind] = -1 * rho2;
        }
    }
    Scaling<real_t> B(VolumeDescriptor{{n, n, layers + 1}}, factorsOfB);

    DataContainer<real_t> c(VolumeDescriptor{{n, n, layers + 1}});
    c = 0;

    Constraint<real_t> constraint(A, B, c);

    // generate circular trajectory
    index_t numAngles{360}, arc{360};
    const auto distance = static_cast<real_t>(n);
    auto sinoDescriptor = LimitedAngleTrajectoryGenerator::createTrajectory(
        numAngles, std::pair(geometry::Degree(60), geometry::Degree(120)), VolumeDescriptor{{n, n}},
        arc, distance * 100.0f, distance);

    SiddonsMethodCUDA projector(VolumeDescriptor{{n, n}}, *sinoDescriptor);

    // simulate noise // TODO try with/without noise later on
    // DataContainer<real_t> noise(projector.getRangeDescriptor());
    // TODO do this ?
    // noise = 0;
    // PoissonNoiseGenerator pNG(0.5f);
    // GaussianNoiseGenerator gNG(0, 0.1f);
    // noise = pNG(noise) + gNG(noise);

    // simulate the sinogram
    auto sinogram = projector.apply(phantom);

    // setup reconstruction problem
    WLSProblem wlsProblem(projector, sinogram);

    DataContainer<real_t> wL1NWeights(VolumeDescriptor{{n, n, layers}});
    wL1NWeights = 0.001f;
    WeightedL1Norm<real_t> weightedL1Norm(LinearResidual<real_t>{shearletTransform}, wL1NWeights);
    RegularizationTerm<real_t> wL1NormRegTerm(1, weightedL1Norm);

    Indicator<real_t> indicator(VolumeDescriptor{{n, n}});
    RegularizationTerm<real_t> indicRegTerm(1, indicator);

    SplittingProblem<real_t> splittingProblem(
        wlsProblem.getDataTerm(), std::vector{wL1NormRegTerm, indicRegTerm}, constraint);

    SHADMM<CG, SoftThresholding, real_t> shadmm(splittingProblem);

    auto sol = shadmm.solve(50);
    printf("square l2 norm of sol. is %f\n", sol.squaredL2Norm());

    // write the solution out
    EDF::write(sol, "shadmm.edf");
}

int main()
{
    try {
        example2d_shadmm();
    } catch (std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << "\n";
    }
}