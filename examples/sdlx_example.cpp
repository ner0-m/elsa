/// Elsa example program: basic 2d X-ray CT simulation and reconstruction

#include "elsa.h"

#include "SHADMM.h"
#include "NoiseGenerators.h"

#include <string>
#include <iostream>
#include <filesystem>

using namespace elsa;

void sdlx_example()
{
    index_t threads = 6;

    std::vector<std::filesystem::path> filePaths;
    std::string srcDir = nullptr;
    std::string destDir = nullptr;
    for (const auto& entry : std::filesystem::directory_iterator(srcDir)) {
        filePaths.emplace_back(entry.path());
    }

#pragma omp parallel for num_threads(threads)
    for (const auto& filePath : filePaths) {
        index_t n = 256;
        DataContainer<real_t> image = EDF::read(filePath);
        if (n != image.getDataDescriptor().getNumberOfCoefficientsPerDimension()[0]) {
            throw InvalidArgumentError("Different shapes than expected!");
        }

        index_t noIterations = 5;

        // random numbers
        real_t rho1 = 1 / 2;
        real_t rho2 = 1; /// LtI fixes this to 1

        /// AT = (ρ_1*SH^T, ρ_2*I_n^2 ) ∈ R ^ n^2 × (L+1)n^2
        ShearletTransform<real_t> shearletTransform(n, n);
        index_t L = shearletTransform.getL();

        IndexVector_t slicesInBlock(2);
        slicesInBlock << L, 1;

        std::vector<std::unique_ptr<LinearOperator<real_t>>> opsOfA(0);
        Scaling<real_t> scaling(VolumeDescriptor{{n, n}}, rho2);
        opsOfA.push_back(shearletTransform.clone()); // TODO mult. with rho1 later on
        opsOfA.push_back(scaling.clone());
        BlockLinearOperator<real_t> A{
            VolumeDescriptor{{n, n}},
            PartitionDescriptor{VolumeDescriptor{{n, n, L + 1}}, slicesInBlock}, opsOfA,
            BlockLinearOperator<real_t>::BlockType::ROW};

        /// B = diag(−ρ_1*1_Ln^2 , −ρ_2*1_n^2) ∈ R ^ (L+1)n^2 × (L+1)n^2
        DataContainer<real_t> factorsOfB(VolumeDescriptor{n, n, L + 1});
        for (int ind = 0; ind < factorsOfB.getSize(); ++ind) {
            if (ind < (n * n * L)) {
                factorsOfB[ind] = -1 * rho1;
            } else {
                factorsOfB[ind] = -1 * rho2;
            }
        }
        Scaling<real_t> B(VolumeDescriptor{{n, n, L + 1}}, factorsOfB);

        DataContainer<real_t> dCC(VolumeDescriptor{{n, n, L + 1}});
        dCC = 0;

        Constraint<real_t> constraint(A, B, dCC);

        DataContainer<real_t> ones(VolumeDescriptor{{n, n, L}});
        ones = 0.001f;
        WeightedL1Norm<real_t> weightedL1Norm(LinearResidual<real_t>{shearletTransform}, ones);
        RegularizationTerm<real_t> wL1NormRegTerm(1, weightedL1Norm);

        Indicator<real_t> indicator(VolumeDescriptor{{n, n}});
        RegularizationTerm<real_t> indicRegTerm(1, indicator);

        // generate circular trajectory
        index_t numAngles{360}, arc{360};
        const auto distance = static_cast<real_t>(n);
        auto sinoDescriptor = LimitedAngleTrajectoryGenerator::createTrajectory(
            numAngles, std::pair(geometry::Degree(75), geometry::Degree(105)),
            VolumeDescriptor{{n, n}}, arc, distance * 100.0f, distance);

        SiddonsMethodCUDA projector(VolumeDescriptor{{n, n}}, *sinoDescriptor);

        // simulate noise // TODO try with/without noise later on
        // DataContainer<real_t> noise(projector.getRangeDescriptor());
        // TODO do this ?
        // noise = 0;
        // PoissonNoiseGenerator pNG(0.5f);
        // GaussianNoiseGenerator gNG(0, 0.1f);
        // noise = pNG(noise) + gNG(noise);

        // simulate the sinogram
        auto sinogram = projector.apply(image);

        // setup reconstruction problem
        WLSProblem wlsProblem(projector, sinogram);

        SplittingProblem<real_t> splittingProblem(
            wlsProblem.getDataTerm(), std::vector{wL1NormRegTerm, indicRegTerm}, constraint);

        SHADMM<CG, SoftThresholding, real_t> shadmm(splittingProblem);

        auto sol = shadmm.solve(noIterations);
        printf("square l2 norm of sol. is %f\n", sol.squaredL2Norm());

        // write the solution out
        EDF::write(sol, destDir.append(filePath.filename().string()));
    }
}

int main()
{
    try {
        sdlx_example();
    } catch (std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << "\n";
    }
}
