#include "SplittingProblem.h"
#include "CG.h"
#include "SoftThresholding.h"
#include "SHADMM.h"
#include "ConeAdaptedDiscreteShearletTransform.h"

namespace elsa
{
    class BootstrappingShearletAndDLModel
    {
        void initialShearletTest()
        {
            // 1. generate first discrete shearlet system
            // 2. apply discrete shearlet transform to a given benchmarking image (maybe an
            // outsourced one by Mayo C.)
            // 3. display coefficients alongside the aforementioned image
        }

        void runHybrid(LinearOperator<real_t> R, DataContainer<real_t> eta)
        {
            // TODO what exactly is R (radon operator) and eta (measurement noise) here
            // 1. solve the the LASSO problem fStar = 1/2||Rf - y||^2_2 + lambda*||SH(f)||_1
            //       here ADMM can be used (ISTA/FISTA and maybe others as well)

            WLSProblem<real_t> wlsProb(R, eta);
            L1Norm<real_t> l1Norm(eta.getDataDescriptor());    // or maybe some other DataDescriptor
            RegularizationTerm<real_t> regTerm(1 / 2, l1Norm); // regTerm(lambda, l1Norm)
            // SplittingProblem<real_t> splittingProblem(wlsProb.getDataTerm(), regTerm, );
            // SHADMM<CG, SoftThresholding> admmsh(splittingProblem);
            // TODO is fStar actually a LASSO problem? The regularization term is not the typical
            //  ||f||_1 but ||SH(f)||_1.

            // TODO how to best test wavelets? Similarly to how it is done with
            // wavelets/curvelets?

            // 2. apply the trained DL model to f* [through trainDLModel()]
            //       myModel = models.myModel(preTrained=True)
            //       F = myModel.pred(ConeAdaptedDiscreteShearletTransform<real_t>.transform(fStar))

            // 3. apply inverse shearlet transform to sum of visible coefficients and DL output
            //   to get the final output (image)
            // auto sumInvAndVis = F + ConeAdaptedDiscreteShearletTransform<real_t>::apply(fStar)
            // DataContainer<real_t> finalResult =
            // ConeAdaptedDiscreteShearletTransform<real_t>::applyInverse(sumInvAndVis)
        }

        void trainDLModel()
        {
            // 1. prepare data (do I need to "corrupt" data as in limited angle scans?)
            // 2. train, try a myriad of hyperparameters?
            // 2. save an accurate (enough?) model?
        }
    };
} // namespace elsa
