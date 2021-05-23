#include "SplittingProblem.h"
#include "CG.h"
#include "SoftThresholding.h"
#include "WeightedL1Norm.h"
#include "SHADMM.h"
#include "Identity.h"
#include "ConeAdaptedDiscreteShearletTransform.h"

namespace elsa
{
    /// class that describes current projected goals for shearlets and deep learning in x-ray ct
    class BootstrappingSDLX
    {
        void initialShearletTest()
        {
            // 1. generate first discrete shearlet system
            // 2. apply discrete shearlet transform to a given benchmarking image (maybe an
            // outsourced one by Mayo C.)
            // 3. display coefficients alongside the aforementioned image
        }

        void runHybrid(LinearOperator<real_t> R, DataContainer<real_t> y,
                       ConeAdaptedDiscreteShearletTransform<real_t> shearletTransform,
                       DataContainer<real_t> weights)
        {
            // TODO what exactly is R (radon operator) and y (measurement noise) here

            /// 1. solve the the LASSO problem fStar = 1/2||Rf - y||^2_2 + lambda*||SH(f)||_1,w
            ///  here ADMM can be used (what else can be used here?)
            /// building now the splitting problem of the above
            WLSProblem<real_t> wlsProb(R, y);
            WeightedL1Norm<real_t> wl1Norm(LinearResidual<real_t>{shearletTransform}, weights);
            RegularizationTerm<real_t> wl1RegTerm(1 / 2, wl1Norm);
            // indicator if P2z is >=0, P2z ∈ R ^ n^2
            Indicator<real_t> indicator(R.getRangeDescriptor());
            std::vector<RegularizationTerm<data_t>> regTerms{wl1RegTerm, indicator};
            // AT = (ρ_1*SH^T, ρ_2*I_n2 ) ∈ R ^ n^2 × (J+1)n^2
            BlockLinearOperator<real_t> A(
                BlockLinearOperator<real_t>::OperatorList(
                    rho1 * ConeAdaptedDiscreteShearletTransform<real_t>(...),
                    Scaling<real_t>(rho2)),
                BlockLinearOperator<real_t>::BlockType.COL);
            // B = diag(−ρ_1*1_Jn^2 , −ρ_2*1_n2) ∈ R ^ (J+1)n2 × (J+1)n^2
            BlockLinearOperator<real_t> B(
                BlockLinearOperator<real_t>::OperatorList(-rho1 * Identity<real_t>(...),
                                                          -rho2 * Identity<real_t>(...)),
                BlockLinearOperator<real_t>::BlockType.COL);
            DataContainer<real_t> zeroes(A.domainDescriptor);
            zeroes = 0;
            Constraint<real_t> constraint(A, B, zeroes);
            SplittingProblem<real_t> splittingProblem(wlsProb.getDataTerm(), regTerms, constraint);
            SHADMM<CG> shadmm(splittingProblem);

            // TODO how to best test shearlets? Similarly to how it is done with wavelets/curvelets?

            /// 2. apply the trained DL model to f* [through trainDLModel()]
            auto myModel = models.myModel(preTrained = True);
            auto F = myModel.pred(shearletTransform.transform(fStar));

            /// 3. apply inverse shearlet transform to sum of visible coefficients and DL output
            ///  to get the final output (image)
            auto sumInvAndVis = F + shearletTransform.apply(fStar);
            DataContainer<real_t> hybridOutput = shearletTransform.applyAdjoint(sumInvAndVis);
        }

        void createModel()
        {
            /// example model taken from the docs, build a U-Net model
            auto model = ml::Sequential(ml::Input(inputDesc, /* batch-size */ 1),
                                        ml::Dense(128, ml::Activation::Relu),
                                        ml::Dense(10, ml::Activation::Relu), ml::Softmax());
            // Define an Adam optimizer
            auto opt = ml::Adam();
            // Compile the model
            model.compile(ml::SparseCategoricalCrossentropy(), &opt);
            model.fit(inputs, labels, /* epochs */ 10);

            /// details about modified U-Net below
            /// NN: R ^ J x n x n -> R ^ J x n x n
            /// fully convolutional neural network

            // TODO consider adding TDB, TD, TU for a modification of U-Net

            // additional model to try given sufficient time: TransUNet
        }

        void trainDLModel()
        {
            // 1. prepare data (do I need to "corrupt" data as in limited angle scans?)
            // 2. train, try a myriad of hyperparameters?
            // 2. save an accurate (enough?) model? TODO how to save a model in elsa ml?

            // TODO to simply create a new model, create a new class? where should it be stored?
        }
    };
} // namespace elsa
