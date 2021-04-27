#include "SplittingProblem.h"
#include "ADMM.h"
#include "CG.h"
#include "SoftThresholding.h"

namespace elsa
{
    class BootstrappingShearletAndDLModel
    {
        void runHybrid()
        {
            // TODO what exactly is R (radon operator) and y (measurement noise) here
            // 1. solve the the LASSO problem fStar = 1/2||Rf - y||^2_2 + lambda*||SH(f)||_1
            //       here ADMM can be used (ISTA/FISTA and maybe others as well)

            //       WLSProblem<real_t> wlsProb(R, y);
            //       L1Norm<real_t> l1Norm(dataDescriptor);
            //       RegularizationTerm<real_t> regTerm(lambda, l1Norm);
            //       SplittingProblem<real_t> splittingProblem(wlsProb.getDataTerm(), regTerm);
            //       ADMM<CG, SoftThresholding> admm(splittingProblem);
            // TODO is fStar actually a LASSO problem? The regularization term is not the typical
            //  ||f||_1 but ||SH(f)||_1.

            // TODO how to best test shearlets? Similarly to how it is done with wavelets/curvelets?

            // 2. apply the trained DL model to f*
            //       myModel = models.myModel(preTrained=True)
            //       F = myModel.pred(ConeAdaptedDiscreteShearlet.transform(fStar))

            // 3. apply inverse shearlet transform to sum of visible coefficients and DL output
            //   to get the final output (image)
            //       sumInvAndVis = F + ConeAdaptedDiscreteShearlet.transform(fStar)
            //       finalResult = ConeAdaptedDiscreteShearlet.inverseTransform(sumInvAndVis)
        }

        void trainDLModel()
        {
            // 1. prepare data (do I need to "corrupt" data as in limited angle scans?)
            // 2. train, try a myriad of hyperparameters?
            // 2. save an accurate (enough?) model?
        }
    };
} // namespace elsa
