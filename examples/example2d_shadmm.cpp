/// Elsa example program: basic 2d X-ray CT simulation and reconstruction

#include "elsa.h"

#include "SHADMM.h"
#include "VolumeDescriptor.h"
#include "BlockLinearOperator.h"
#include "CG.h"
#include "Identity.h"
#include "FISTA.h"
#include "Logger.h"

#include <iostream>

using namespace elsa;

void example2d_shadmm()
{
    // generate 2d phantom
    IndexVector_t size(2);
    size << 128, 128;
    auto phantom = PhantomGenerator<real_t>::createModifiedSheppLogan(size);

    // random numbers
    real_t rho1 = 23;
    real_t rho2 = 86;

    /// AT = (ρ_1*SH^T, ρ_2*I_n^2 ) ∈ R ^ n^2 × (L+1)n^2
    ShearletTransform<real_t> shearletTransform(size);
    index_t L = shearletTransform.getL();
    index_t n = shearletTransform.getWidth(); // size[0] == size[1], for now at least

    std::vector<std::unique_ptr<LinearOperator<real_t>>> opsOfA(0);
    Scaling<real_t> scaling1(VolumeDescriptor{n * n}, rho2);
    opsOfA.push_back(std::move(shearletTransform.clone())); // TODO mult. with rho1 later on
    opsOfA.push_back(std::move(scaling1.clone()));
    BlockLinearOperator<real_t> A{opsOfA, BlockLinearOperator<real_t>::BlockType::ROW};

    /// B = diag(−ρ_1*1_Ln^2 , −ρ_2*1_n^2) ∈ R ^ (L+1)n^2 × (L+1)n^2
    DataContainer<real_t> factorsOfB(VolumeDescriptor{(L + 1) * n * n});
    for (int ind = 0; ind < factorsOfB.getSize(); ++ind) {
        if (ind < (L * n * n)) {
            factorsOfB[ind] = -1 * rho1;
        } else {
            factorsOfB[ind] = -1 * rho2;
        }
    }
    Scaling<real_t> B(VolumeDescriptor{(L + 1) * n * n}, factorsOfB);

    DataContainer<real_t> dCC(VolumeDescriptor{(L + 1) * n * n});
    dCC = 0;

    Constraint<real_t> constraint(A, B, dCC);

    // generate circular trajectory
    index_t numAngles{180}, arc{360};
    const auto distance = static_cast<real_t>(size(0));
    auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(
        numAngles, VolumeDescriptor{{n, n}}, arc, distance * 100.0f, distance);

    SiddonsMethod projector(VolumeDescriptor{{n, n}}, *sinoDescriptor);

    // simulate no noise // TODO try with noise later on
    DataContainer<real_t> noNoise(projector.getRangeDescriptor());
    noNoise = 0;

    // setup reconstruction problem
    WLSProblem wlsProblem(projector, noNoise);

    DataContainer<real_t> ones(VolumeDescriptor{L * n * n});
    ones = 1;
    WeightedL1Norm<real_t> weightedL1Norm(ones);
    RegularizationTerm<real_t> wL1NormRegTerm(1, weightedL1Norm);

    Indicator<real_t> indicator(VolumeDescriptor{n * n});
    RegularizationTerm<real_t> indicRegTerm(1, indicator);

    SplittingProblem<real_t> splittingProblem(
        wlsProblem.getDataTerm(), std::vector{wL1NormRegTerm, indicRegTerm}, constraint);

    SHADMM<CG, SoftThresholding, real_t> shadmm(splittingProblem);

    auto sol = shadmm.solve(10);
    sol = sol.viewAs(VolumeDescriptor{{n, n}});
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
