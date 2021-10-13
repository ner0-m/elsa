#include "InpaintMissingSingularitiesTask.h"
#include "NoiseGenerators.h"
#include "CircleTrajectoryGenerator.h"
#include "LimitedAngleTrajectoryGenerator.h"
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
#include "ADMM.h"
#include "SiddonsMethod.h"
#include "Logger.h"

#ifdef ELSA_CUDA_PROJECTORS
#include "SiddonsMethodCUDA.h"
#include "JosephsMethodCUDA.h"
#endif

#include <iostream>
#include <utility>

namespace elsa
{
    template <typename data_t>
    DataContainer<data_t>
        InpaintMissingSingularitiesTask<data_t>::reconstructVisibleCoeffsOfLimitedAngleCT(
            DataContainer<data_t> image,
            std::pair<elsa::geometry::Degree, elsa::geometry::Degree> missingWedgeAngles,
            index_t numOfAngles, index_t arc, index_t solverIterations, data_t rho1, data_t rho2)
    {
        const DataDescriptor& dataDescriptor = image.getDataDescriptor();

        if (dataDescriptor.getNumberOfDimensions() != 2) {
            throw new InvalidArgumentError(
                "InpaintLimitedAngleSingularitiesTask: only 2D images are supported");
        }

        index_t width = dataDescriptor.getNumberOfCoefficientsPerDimension()[0];
        index_t height = dataDescriptor.getNumberOfCoefficientsPerDimension()[1];
        ShearletTransform<data_t, data_t> shearletTransform(width, height);
        index_t layers = shearletTransform.getNumOfLayers();

        /// AT = (ρ_1*SH^T, ρ_2*I_n^2 ) ∈ R ^ n^2 × (L+1)n^2
        std::vector<std::unique_ptr<LinearOperator<data_t>>> opsOfA(0);
        Scaling<data_t> scaling(VolumeDescriptor{{width, height}}, rho2);
        opsOfA.push_back(std::move(shearletTransform.clone())); // TODO mult. with rho1 later on
        opsOfA.push_back(std::move(scaling.clone()));
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

        DataContainer<data_t> c(VolumeDescriptor{{width, height, layers + 1}});
        c = 0;

        Constraint<data_t> constraint(A, B, c);

        // generate limited-angle trajectory
        const auto distance = static_cast<real_t>(width);
        auto sinoDescriptor = LimitedAngleTrajectoryGenerator::createTrajectory(
            numOfAngles, missingWedgeAngles, VolumeDescriptor{{width, height}}, arc,
            distance * 100.0f, distance);

#ifdef ELSA_CUDA_PROJECTORS
        SiddonsMethodCUDA<data_t> projector(VolumeDescriptor{{width, height}}, *sinoDescriptor);
#else
        SiddonsMethod<data_t> projector(VolumeDescriptor{{width, height}}, *sinoDescriptor);
#endif

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

        // simulate the limited-angle sinogram
        DataContainer<data_t> limitedAngleSinogram = projector.apply(image);

        // setup reconstruction problem
        WLSProblem<data_t> wlsProblem(projector, limitedAngleSinogram);

        DataContainer<data_t> wL1NWeights(VolumeDescriptor{{width, height, layers}});
        wL1NWeights = 0.001f;
        WeightedL1Norm<data_t> weightedL1Norm(LinearResidual<data_t>{shearletTransform},
                                              wL1NWeights);
        RegularizationTerm<data_t> wL1NormRegTerm(1, weightedL1Norm);

        Indicator<data_t> indicator(VolumeDescriptor{{width, height}});
        RegularizationTerm<data_t> indicRegTerm(1, indicator);

        SplittingProblem<data_t> splittingProblem(
            wlsProblem.getDataTerm(), std::vector{wL1NormRegTerm, indicRegTerm}, constraint);

        // solve the reconstruction problem
        ADMM<CG, SoftThresholding, data_t> admm(splittingProblem, true);

        Logger::get("Info")->info("Solving reconstruction using {} iterations of ADMM",
                                  solverIterations);
        return admm.solve(solverIterations);
    }

    template <typename data_t>
    DataContainer<data_t>
        InpaintMissingSingularitiesTask<data_t>::reconstructVisibleCoeffsOfSparseCT(
            DataContainer<data_t> image, index_t numOfAngles, index_t arc, index_t solverIterations,
            data_t rho1, data_t rho2)
    {
        const DataDescriptor& dataDescriptor = image.getDataDescriptor();

        if (dataDescriptor.getNumberOfDimensions() != 2) {
            throw new InvalidArgumentError(
                "InpaintLimitedAngleSingularitiesTask: only 2D images are supported");
        }

        index_t width = dataDescriptor.getNumberOfCoefficientsPerDimension()[0];
        index_t height = dataDescriptor.getNumberOfCoefficientsPerDimension()[1];
        ShearletTransform<data_t, data_t> shearletTransform(width, height);
        index_t layers = shearletTransform.getNumOfLayers();

        /// AT = (ρ_1*SH^T, ρ_2*I_n^2 ) ∈ R ^ n^2 × (L+1)n^2
        std::vector<std::unique_ptr<LinearOperator<data_t>>> opsOfA(0);
        Scaling<data_t> scaling(VolumeDescriptor{{width, height}}, rho2);
        opsOfA.push_back(std::move(shearletTransform.clone())); // TODO mult. with rho1 later on
        opsOfA.push_back(std::move(scaling.clone()));
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

        DataContainer<data_t> c(VolumeDescriptor{{width, height, layers + 1}});
        c = 0;

        Constraint<data_t> constraint(A, B, c);

        // generate limited-angle trajectory
        const auto distance = static_cast<real_t>(width);
        auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(
            numOfAngles, VolumeDescriptor{{width, height}}, arc, distance * 100.0f, distance);

#ifdef ELSA_CUDA_PROJECTORS
        SiddonsMethodCUDA<data_t> projector(VolumeDescriptor{{width, height}}, *sinoDescriptor);
#else
        SiddonsMethod<data_t> projector(VolumeDescriptor{{width, height}}, *sinoDescriptor);
#endif

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

        // simulate the sparse sinogram
        DataContainer<data_t> sparseSinogram = projector.apply(image);

        // setup reconstruction problem
        WLSProblem<data_t> wlsProblem(projector, sparseSinogram);

        DataContainer<data_t> wL1NWeights(VolumeDescriptor{{width, height, layers}});
        wL1NWeights = 0.001f;
        WeightedL1Norm<data_t> weightedL1Norm(LinearResidual<data_t>{shearletTransform},
                                              wL1NWeights);
        RegularizationTerm<data_t> wL1NormRegTerm(1, weightedL1Norm);

        Indicator<data_t> indicator(VolumeDescriptor{{width, height}});
        RegularizationTerm<data_t> indicRegTerm(1, indicator);

        SplittingProblem<data_t> splittingProblem(
            wlsProblem.getDataTerm(), std::vector{wL1NormRegTerm, indicRegTerm}, constraint);

        // solve the reconstruction problem
        ADMM<CG, SoftThresholding, data_t> admm(splittingProblem, true);

        Logger::get("Info")->info("Solving reconstruction using {} iterations of ADMM",
                                  solverIterations);
        return admm.solve(solverIterations);
    }

    //    template <typename data_t>
    //    void InpaintMissingSingularitiesTask<data_t>::trainPhantomNet(
    //        const std::vector<DataContainer<data_t>>& x, const std::vector<DataContainer<data_t>>&
    //        y, index_t epochs)
    //    {
    //        if (x.size() != y.size()) {
    //            throw new LogicError("InpaintMissingSingularitiesTask: sizes of the x
    //            (inputs) "
    //                                 "and y (labels) variables for training the PhantomNet must
    //                                 match");
    //        }
    //
    //        // define an Adam optimizer
    //        auto opt = ml::Adam();
    //
    //        // compile the model
    //        phantomNet.compile(ml::SparseCategoricalCrossentropy(), &opt);
    //
    //        // train the model
    //        phantomNet.fit(x, y, epochs);
    //    }

    template <typename data_t>
    DataContainer<data_t>
        InpaintMissingSingularitiesTask<data_t>::combineVisCoeffsToInpaintedInvisCoeffs(
            DataContainer<data_t> visCoeffs, DataContainer<data_t> invisCoeffs)
    {
        if (visCoeffs.getDataDescriptor() != invisCoeffs.getDataDescriptor()) {
            throw new LogicError("InpaintMissingSingularitiesTask: DataDescriptors of the "
                                 "visible coefficients and invisible coefficients must match");
        }

        index_t width = visCoeffs.getDataDescriptor().getNumberOfCoefficientsPerDimension()[0];
        index_t height = visCoeffs.getDataDescriptor().getNumberOfCoefficientsPerDimension()[1];

        if (width != height) {
            throw new InvalidArgumentError(
                "InpaintMissingSingularitiesTask: only 2D images are supported");
        }

        ShearletTransform<data_t, data_t> shearletTransform(width, height);

        // combine the coefficients
        return shearletTransform.applyAdjoint(visCoeffs + invisCoeffs);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class InpaintMissingSingularitiesTask<float>;
    template class InpaintMissingSingularitiesTask<double>;
} // namespace elsa
