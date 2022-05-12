#pragma once

#include "elsaDefines.h"
#include "DataContainer.h"
#include "StrongTypes.h"
#include "Scaling.h"
#include "WLSProblem.h"
#include "Constraint.h"
#include "RegularizationTerm.h"
#include "CG.h"
#include "SoftThresholding.h"
#include "SiddonsMethod.h"
#include "ADMM.h"
#include "../ml/Common.h"
#include "../ml/Model.h"
#include "CircleTrajectoryGenerator.h"

#ifdef ELSA_CUDA_PROJECTORS
#include "SiddonsMethodCUDA.h"
#include "JosephsMethodCUDA.h"
#endif

namespace elsa::task
{
    /**
     * @brief task representing the construction of trajectories given various parameters
     *
     * This task employs the Builder pattern in order the address the growing number of setup
     * parameters. It can currently construct trajectories for the LimitedAngleTrajectoryGenerator
     * and CircleTrajectoryGenerator.
     *
     * @author Andi Braimllari - initial code
     *
     * // TODO
     * @tparam data_t data type for the domain and range of the trajectory construction task,
     * defaulting to real_t
     */
    namespace ConstructTrajectory
    {
        class LimitedAngleTrajectoryBuilder
        {
        private:
            index_t _numberOfPoses;
            std::unique_ptr<DataDescriptor> _volumeDescriptor;
            index_t _arcDegrees;
            float _sourceToCenter;
            float _centerToDetector;

            std::pair<elsa::geometry::Degree, elsa::geometry::Degree> _missingWedgeAngles;
            bool _mirrored;

        public:
            LimitedAngleTrajectoryBuilder& setNumberOfPoses(index_t numberOfPoses)
            {
                _numberOfPoses = numberOfPoses;
                return *this;
            }

            LimitedAngleTrajectoryBuilder& setMissingWedgeAngles(
                std::pair<elsa::geometry::Degree, elsa::geometry::Degree> missingWedgeAngles)
            {
                _missingWedgeAngles = missingWedgeAngles;
                return *this;
            }

            LimitedAngleTrajectoryBuilder& setArcDegrees(index_t arcDegrees)
            {
                _arcDegrees = arcDegrees;
                return *this;
            }

            LimitedAngleTrajectoryBuilder& setSourceToCenter(float sourceToCenter)
            {
                _sourceToCenter = sourceToCenter;
                return *this;
            }

            LimitedAngleTrajectoryBuilder& setCenterToDetector(float centerToDetector)
            {
                _centerToDetector = centerToDetector;
                return *this;
            }

            LimitedAngleTrajectoryBuilder& setMirrored(bool mirrored)
            {
                _mirrored = mirrored;
                return *this;
            }

            std::unique_ptr<DetectorDescriptor> build(const DataDescriptor& volumeDescriptor)
            {
                return LimitedAngleTrajectoryGenerator::createTrajectory(
                    _numberOfPoses, _missingWedgeAngles, volumeDescriptor, _arcDegrees,
                    _sourceToCenter, _centerToDetector, _mirrored);
            }
        };

        class CircleTrajectoryBuilder
        {
        private:
            index_t _numberOfPoses;
            index_t _arcDegrees;
            float _sourceToCenter;
            float _centerToDetector;

        public:
            CircleTrajectoryBuilder& setNumberOfPoses(index_t numberOfPoses)
            {
                _numberOfPoses = numberOfPoses;
                return *this;
            }

            CircleTrajectoryBuilder& setArcDegrees(index_t arcDegrees)
            {
                _arcDegrees = arcDegrees;
                return *this;
            }

            CircleTrajectoryBuilder& setSourceToCenter(float sourceToCenter)
            {
                _sourceToCenter = sourceToCenter;
                return *this;
            }

            CircleTrajectoryBuilder& setCenterToDetector(float centerToDetector)
            {
                _centerToDetector = centerToDetector;
                return *this;
            }

            std::unique_ptr<DetectorDescriptor> build(const DataDescriptor& volumeDescriptor)
            {
                return CircleTrajectoryGenerator::createTrajectory(_numberOfPoses, volumeDescriptor,
                                                                   _arcDegrees, _sourceToCenter,
                                                                   _centerToDetector);
            }
        };

        LimitedAngleTrajectoryBuilder limitedAngle = LimitedAngleTrajectoryBuilder();
        CircleTrajectoryBuilder circle =
            CircleTrajectoryBuilder(); // TODO would circular be a better name?

    } // namespace ConstructTrajectory

    /**
     * @brief task representing inpainting of missing singularities (coefficients) in
     * limited-angle or sparse-view CT scans
     *
     * This task offers three functionalities,
     * 1. Based on a given image, generate a limited-angle/sparse sinogram of it, and reconstruct
     * the original image using an l1-regularization with shearlets. ADMM is chosen as the solver
     * here.
     * 2. Train a fully convolutional NN model (by default PhantomNet) to learn intrinsic edge
     * patterns of natural-images, so that it can later properly infer the missing data in
     * limited-angle scans, in the Fourier domain.
     * 3. Combine the visible coefficients with the learned invisible coefficients inferred by
     * PhantomNet, i.e. apply the inverse shearlet transform to the sum of the direct shearlet
     * transforms of the l1-regularization with shearlets and shearlet transformed PhantomNet
     * output. This aims to tackle the limited-angle/sparse problem, thus generating a
     * reconstruction as close to the ground truth as possible, which actually is the final goal of
     * this task.
     *
     * Note that, if one requires to have finer-grained control over various elements and aspects of
     * this tasks, feel free to use the already-available relevant classes. This task exists to
     * offer ease of use in tackling the limited-angle/sparse problem through this hybrid approach.
     * For additional details, please refer to the references section.
     *
     * @author Andi Braimllari - initial code
     *
     * @tparam data_t data type for the domain and range of the SDLX task, defaulting to real_t
     *
     * References:
     * TODO
     * https://arxiv.org/pdf/1811.04602.pdf
     */
    namespace InpaintMissingSingularities
    {
        /// Reconstructs the l1-regularization with shearlets by utilizing ADMM. By default the CUDA
        /// projector will be used if available.
        template <typename data_t = real_t>
        DataContainer<data_t>
            reconstructVisibleCoeffs(DataContainer<data_t> image,
                                     std::unique_ptr<DetectorDescriptor> trajectory,
                                     index_t solverIterations = 50, data_t rho1 = 1.0 / 2,
                                     data_t rho2 = 1) // DataContainer<data_t> noise = 0,
        {
            const DataDescriptor& domainDescriptor = image.getDataDescriptor();

            if (domainDescriptor.getNumberOfDimensions() != 2) {
                throw new InvalidArgumentError(
                    "task::InpaintMissingSingularities: only 2D images are supported");
            }

            index_t width = domainDescriptor.getNumberOfCoefficientsPerDimension()[0];
            index_t height = domainDescriptor.getNumberOfCoefficientsPerDimension()[1];
            ShearletTransform<data_t, data_t> shearletTransform(width, height);
            index_t numOfLayers = shearletTransform.getNumOfLayers();

            VolumeDescriptor numOfLayersPlusOneDescriptor{{width, height, numOfLayers + 1}};

            /// AT = (ρ_1*SH^T, ρ_2*I_n^2 ) ∈ R ^ n^2 × (L+1)n^2
            IndexVector_t slicesInBlock(2);
            slicesInBlock << numOfLayers, 1;
            std::vector<std::unique_ptr<LinearOperator<real_t>>> opsOfA(0);
            Scaling<real_t> scaling(domainDescriptor, rho2);
            opsOfA.push_back((rho1 * shearletTransform).clone()); // TODO double check
            opsOfA.push_back(scaling.clone());
            BlockLinearOperator<real_t> A(
                domainDescriptor, PartitionDescriptor{numOfLayersPlusOneDescriptor, slicesInBlock},
                opsOfA, BlockLinearOperator<real_t>::BlockType::ROW);

            /// B = diag(−ρ_1*1_Ln^2, −ρ_2*1_n^2) ∈ R ^ (L+1)n^2 × (L+1)n^2
            DataContainer<data_t> factorsOfB(numOfLayersPlusOneDescriptor);
            for (int ind = 0; ind < factorsOfB.getSize(); ++ind) {
                if (ind < (numOfLayers * width * height)) {
                    factorsOfB[ind] = -1 * rho1;
                } else {
                    factorsOfB[ind] = -1 * rho2;
                }
            }
            Scaling<data_t> B(numOfLayersPlusOneDescriptor, factorsOfB);

            DataContainer<data_t> c(numOfLayersPlusOneDescriptor);
            c = 0;

            Constraint<data_t> constraint(A, B, c);

#ifdef ELSA_CUDA_PROJECTORS
            SiddonsMethodCUDA<data_t> projector(downcast<VolumeDescriptor>(domainDescriptor),
                                                *trajectory);
#else
            SiddonsMethod<data_t> projector(downcast<VolumeDescriptor>(domainDescriptor),
                                            *trajectory);
#endif

            // simulate the defined sinogram
            DataContainer<data_t> sinogram = projector.apply(image);

            // sinogram += noise;

            // setup reconstruction problem
            WLSProblem<data_t> wlsProblem(projector, sinogram);

            DataContainer<data_t> wL1NWeights(VolumeDescriptor{{width, height, numOfLayers}});
            wL1NWeights = 0.001f;
            WeightedL1Norm<data_t> weightedL1Norm(LinearResidual<data_t>{shearletTransform},
                                                  wL1NWeights);
            RegularizationTerm<data_t> wL1NormRegTerm(1, weightedL1Norm);

            SplittingProblem<real_t> splittingProblem(wlsProblem.getDataTerm(), wL1NormRegTerm,
                                                      constraint);

            // solve the reconstruction problem
            ADMM<CG, SoftThresholding, data_t> admm(splittingProblem, true);

            Logger::get("Info")->info("Solving reconstruction using {} iterations of ADMM",
                                      solverIterations);

            return admm.solve(solverIterations);
        }

        /// Train a model (by default PhantomNet) to be able to learn the invisible coefficients
        /// in the Fourier domain.
        // TODO not sure if elsa ml can properly train fully convolutional NNs, if not I'd prefer to
        //  keep this, but commented out
        template <typename data_t = real_t, elsa::ml::MlBackend Backend = ml::MlBackend::Auto>
        ml::Model<data_t, Backend>
            inpaintInvisibleCoeffs(const std::vector<DataContainer<data_t>>& x,
                                   const std::vector<DataContainer<data_t>>& y,
                                   //                                   ml::Model<data_t, Backend>
                                   //                                   model = ml::PhantomNet(),
                                   ml::Optimizer<data_t> optimizer = ml::Adam(),
                                   ml::Loss<data_t> loss = ml::SparseCategoricalCrossentropy(),
                                   index_t epochs = 20)
        {
            ml::Model<data_t, Backend> model;
            if (x.size() != y.size()) {
                throw new InvalidArgumentError(
                    "task::InpaintMissingSingularities: sizes of the x (inputs) and y "
                    "(labels) variables for training the model must match");
            }

            if (model.getInputs().front()->getInputDescriptor()
                != model.getOutputs().back()->getOutputDescriptor()) {
                throw new InvalidArgumentError(
                    "task::InpaintMissingSingularities: the provided architecture has different "
                    "shapes for the input and the output");
            }

            // TODO ensure here that the input and output of the model are (L, W, H)

            // compile the model
            model.compile(loss, &optimizer);

            // train the model
            model.fit(x, y, epochs);

            return model;
        }

        /// Simply adds both the coefficients and applies the inverse shearlet transform to it. A
        /// new ShearletTransform object will be created, therefore its spectra has to be
        /// recomputed. The @p visCoeffs will be generated by applying the shearlet transform to the
        /// output of the model-based reconstruction, and the @p invisCoeffs will be the prediction
        /// of the model.
        template <typename data_t = real_t>
        DataContainer<data_t>
            combineVisCoeffsWithInpaintedInvisCoeffs(DataContainer<data_t> visCoeffs,
                                                     DataContainer<data_t> invisCoeffs)
        {
            if (visCoeffs.getDataDescriptor() != invisCoeffs.getDataDescriptor()) {
                throw new InvalidArgumentError(
                    "task::InpaintMissingSingularities: DataDescriptors of the "
                    "visible coefficients and invisible coefficients must match");
            }

            index_t width = visCoeffs.getDataDescriptor().getNumberOfCoefficientsPerDimension()[0];
            index_t height = visCoeffs.getDataDescriptor().getNumberOfCoefficientsPerDimension()[1];

            if (width != height) {
                throw new InvalidArgumentError(
                    "task::InpaintMissingSingularities: only 2D images are supported");
            }

            ShearletTransform<data_t, data_t> shearletTransform(width, height);

            // combine the coefficients
            return shearletTransform.applyAdjoint(visCoeffs + invisCoeffs);
        }

        /// Method that bootstraps all three steps of the SDLX task. For finer-grained control, call
        /// the specific methods.
        template <typename data_t = real_t, elsa::ml::MlBackend Backend = ml::MlBackend::Auto>
        std::vector<DataContainer<data_t>> run(const std::vector<DataContainer<data_t>>& x,
                                               const std::vector<DataContainer<data_t>>& y,
                                               std::unique_ptr<DetectorDescriptor> trajectory,
                                               const std::vector<DataContainer<data_t>>& images)
        {
            std::vector<DataContainer<data_t>> visibleCoeffs;
            for (DataContainer<data_t> image : x) {
                visibleCoeffs.emplace_back(reconstructVisibleCoeffs(image, trajectory));
            }

            auto trainedModel = inpaintInvisibleCoeffs<data_t, Backend>(x, y);

            std::vector<DataContainer<data_t>> sdlxImages;
            for (DataContainer<data_t> image : images) {
                sdlxImages.emplace_back(
                    task::InpaintMissingSingularities::combineVisCoeffsWithInpaintedInvisCoeffs(
                        visibleCoeffs, trainedModel.predict(image)));
            }

            return sdlxImages;
        }
    } // namespace InpaintMissingSingularities
} // namespace elsa::task
