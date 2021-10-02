#include "InpaintLimitedAngleSingularitiesTask.h"

#include "NoiseGenerators.h"
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
#include "SiddonsMethod.h"
#include "Logger.h"

#include "SHADMM.h"

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
        InpaintLimitedAngleSingularitiesTask<data_t>::reconstructVisibleCoeffsOfLimitedAngleCT(
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
        index_t layers = shearletTransform.getL();

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
        return shadmm.solve(solverIterations);
    }

    template <typename data_t>
    void InpaintLimitedAngleSingularitiesTask<data_t>::trainPhantomNet(
        const std::vector<DataContainer<data_t>>& x, const std::vector<DataContainer<data_t>>& y,
        index_t epochs)
    {
        if (x.size() != y.size()) {
            throw new LogicError("InpaintLimitedAngleSingularitiesTask: sizes of the x (inputs) "
                                 "and y (labels) variables for training the PhantomNet must match");
        }

        // define an Adam optimizer
        auto opt = ml::Adam();

        // compile the model
        phantomNet.compile(ml::SparseCategoricalCrossentropy(), &opt);

        // train the model
        phantomNet.fit(x, y, epochs);
    }

    template <typename data_t>
    DataContainer<data_t>
        InpaintLimitedAngleSingularitiesTask<data_t>::combineVisCoeffsToInpaintedInvisCoeffs(
            DataContainer<data_t> visCoeffs, DataContainer<data_t> invisCoeffs)
    {
        if (visCoeffs.getDataDescriptor() != invisCoeffs.getDataDescriptor()) {
            throw new LogicError("InpaintLimitedAngleSingularitiesTask: DataDescriptors of the "
                                 "visible coefficients and invisible coefficients must match");
        }

        index_t width = visCoeffs.getDataDescriptor().getNumberOfCoefficientsPerDimension()[0];
        index_t height = visCoeffs.getDataDescriptor().getNumberOfCoefficientsPerDimension()[1];

        if (width != height) {
            throw new InvalidArgumentError(
                "InpaintLimitedAngleSingularitiesTask: only 2D images are supported");
        }

        ShearletTransform<data_t, data_t> shearletTransform(width, height);

        // combine the coefficients
        return shearletTransform.applyAdjoint(visCoeffs + invisCoeffs);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class InpaintLimitedAngleSingularitiesTask<float>;
    template class InpaintLimitedAngleSingularitiesTask<double>;
} // namespace elsa
