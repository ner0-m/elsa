#include "InpaintLimitedAngleSingularitiesTask.h"

#include "NoiseGenerators.h"
#include "CircleTrajectoryGenerator.h"
#include "Loss.h"
#include "Optimizer.h"
#include "Scaling.h"
#include "Constraint.h"
#include "VolumeDescriptor.h"
#include "RegularizationTerm.h"
#include "SplittingProblem.h"
#include "CG.h"
#include "SoftThresholding.h"
#include "BlockLinearOperator.h"
#include "SiddonsMethod.h"
#include "EDFHandler.h"
#include "Logger.h"

#include "SHADMM.h"

#include <iostream>
#include <utility>

namespace elsa
{
    template <typename data_t>
    void InpaintLimitedAngleSingularitiesTask<data_t>::generateLimitedAngleSinogram(
        DataContainer<data_t> image,
        std::pair<elsa::geometry::Degree, elsa::geometry::Degree> missingWedgeAngles,
        index_t numOfAngles, index_t arc)
    {
        const DataDescriptor& dataDescriptor = image.getDataDescriptor();

        if (dataDescriptor.getNumberOfDimensions() != 2) {
            throw new LogicError("SHADMMTask: only 2D images are supported");
        }

        // generate limited angle trajectory
        const auto distance =
            static_cast<real_t>(dataDescriptor.getNumberOfCoefficientsPerDimension()[0]);
        auto sinoDescriptor = LimitedAngleTrajectoryGenerator::createTrajectory(
            numOfAngles, missingWedgeAngles, dataDescriptor, arc, distance * 100.0f, distance);

        // setup operator for 2D X-ray transform
        Logger::get("Info")->info("Simulating sinogram using Siddon's method");

        SiddonsMethod projector(dynamic_cast<const VolumeDescriptor&>(dataDescriptor),
                                *sinoDescriptor);

        // simulate the sinogram
        auto sinogram = projector.apply(image);

        return sinogram;
    }

    template <typename data_t>
    void InpaintLimitedAngleSingularitiesTask<data_t>::reconstructOnLimitedAngleSinogram(
        DataContainer<data_t> limitedAngleSinogram, index_t numberOfScales,
        index_t solverIterations, data_t rho1, data_t rho2)
    {
        const DataDescriptor& dataDescriptor = image.getDataDescriptor();

        if (dataDescriptor.getNumberOfDimensions() != 2) {
            throw new LogicError("SHADMMTask: only 2D EDF files are supported");
        }

        index_t width = dataDescriptor.getNumberOfCoefficientsPerDimension()[0];
        index_t height = dataDescriptor.getNumberOfCoefficientsPerDimension()[1];
        ShearletTransform<data_t> shearletTransform(width, height);
        index_t layers = _shearletTransform.getL();

        /// AT = (ρ_1*SH^T, ρ_2*I_n^2 ) ∈ R ^ n^2 × (L+1)n^2
        std::vector<std::unique_ptr<LinearOperator<data_t>>> opsOfA(0);
        Scaling<data_t> scaling1(VolumeDescriptor{{width, height}}, rho2);
        opsOfA.push_back(std::move(shearletTransform.clone())); // TODO mult. with rho1 later on
        opsOfA.push_back(std::move(scaling1.clone()));
        BlockLinearOperator<data_t> A{opsOfA, BlockLinearOperator<data_t>::BlockType::ROW};

        /// B = diag(−ρ_1*1_Ln^2 , −ρ_2*1_n^2) ∈ R ^ (L+1)n^2 × (L+1)n^2
        DataContainer<data_t> factorsOfB(VolumeDescriptor{width * height * (layers + 1)});
        for (int ind = 0; ind < factorsOfB.getSize(); ++ind) {
            if (ind < (layers * width * height)) {
                factorsOfB[ind] = -1 * rho1;
            } else {
                factorsOfB[ind] = -1 * rho2;
            }
        }
        Scaling<data_t> B(VolumeDescriptor{{width, height, layers + 1}}, factorsOfB);

        DataContainer<data_t> dCC(VolumeDescriptor{{width, height, layers + 1}});
        dCC = 0;

        Constraint<data_t> constraint(A, B, dCC);

        // generate circular trajectory
        index_t numAngles{180}, arc{360};
        const auto distance = static_cast<real_t>(width);
        auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(
            numAngles, VolumeDescriptor{{width, height}}, arc, distance * 100.0f, distance);

        // TODO consider enabling SiddonsMethodCUDA
        SiddonsMethod<data_t> projector(VolumeDescriptor{{width, height}}, *sinoDescriptor);

        // simulate noise
        DataContainer<data_t> noise(projector.getRangeDescriptor());
        DataContainer<data_t> poissonNoise(projector.getRangeDescriptor());
        poissonNoise = 0;
        DataContainer<data_t> gaussianNoise(projector.getRangeDescriptor());
        gaussianNoise = 0;
        PoissonNoiseGenerator pNG(0.5f);
        GaussianNoiseGenerator gNG(0, 0.1f);
        noise = pNG(poissonNoise) + gNG(gaussianNoise);
        // limitedAngleSinogram += noise; // TODO try with noise later on

        // setup reconstruction problem
        WLSProblem<data_t> wlsProblem(projector, limitedAngleSinogram);

        DataContainer<data_t> ones(VolumeDescriptor{{width, height, layers}});
        ones = 0.001f;
        WeightedL1Norm<data_t> weightedL1Norm(LinearResidual<data_t>{shearletTransform}, ones);
        RegularizationTerm<data_t> wL1NormRegTerm(1, weightedL1Norm);

        Indicator<data_t> indicator(VolumeDescriptor{{width, height}});
        RegularizationTerm<data_t> indicRegTerm(1, indicator);

        SplittingProblem<data_t> splittingProblem(
            wlsProblem.getDataTerm(), std::vector{wL1NormRegTerm, indicRegTerm}, constraint);

        // solve the reconstruction problem
        SHADMM<CG, SoftThresholding, data_t> shadmm(splittingProblem);

        Logger::get("Info")->info("Solving reconstruction using {} iterations of ADMM",
                                  solverIterations);
        auto solution = shadmm.solve(solverIterations);
        printf("square l2 norm of sol. is %f\n", solution.squaredL2Norm()); // TODO remove me

        return solution;
    }

    template <typename data_t>
    void InpaintLimitedAngleSingularitiesTask<data_t>::trainPhantomNet(
        const std::vector<DataContainer<data_t>>& x, const std::vector<DataContainer<data_t>>& y,
        index_t epochs, index_t batchSize)
    {
        // Define an Adam optimizer
        auto opt = ml::Adam();

        // Compile the model
        _phantomNet.compile(ml::SparseCategoricalCrossentropy(), &opt);

        _phantomNet.fit(x, y, epochs);
    }

    template <typename data_t>
    DataContainer<data_t>
        InpaintLimitedAngleSingularitiesTask<data_t>::combineVisCoeffsToInpaintedInvisCoeffs(
            DataContainer<data_t> visCoeffs, DataContainer<data_t> invisCoeffs)
    {
        ShearletTransform<data_t> shearletTransform(n, n);
        return _shearletTransform.applyAdjoint(visCoeffs + invisCoeffs);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class InpaintLimitedAngleSingularitiesTask<float>;
    template class InpaintLimitedAngleSingularitiesTask<double>;
} // namespace elsa
