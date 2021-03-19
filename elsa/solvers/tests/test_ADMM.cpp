#include "ADMM.h"
#include "CG.h"
#include "SoftThresholding.h"
#include "HardThresholding.h"
#include "Identity.h"
#include "FISTA.h"
#include "Logger.h"
#include "VolumeDescriptor.h"

#include <catch2/catch.hpp>
#include <testHelpers.h>

using namespace elsa;

SCENARIO("Solving problems with ADMM")
{
    Logger::setLevel(Logger::LogLevel::WARN);

    GIVEN("some problems and a constraint")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 21, 11;
        VolumeDescriptor volDescr(numCoeff);

        Eigen::Matrix<real_t, Eigen::Dynamic, 1> bVec(volDescr.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer<real_t> dcB(volDescr, bVec);

        Identity<real_t> idOp(volDescr);
        Scaling<real_t> negativeIdOp(volDescr, -1);
        DataContainer<real_t> dCC(volDescr);
        dCC = 0;

        WLSProblem<real_t> wlsProb(idOp, dcB);

        Constraint<real_t> constraint(idOp, negativeIdOp, dCC);

        WHEN("setting up ADMM and FISTA to solve a LASSOProblem")
        {
            L1Norm<real_t> regFunc(volDescr);
            RegularizationTerm<real_t> regTerm(0.000001f, regFunc);

            SplittingProblem<real_t> splittingProblem(wlsProb.getDataTerm(), regTerm, constraint);

            ADMM<CG, SoftThresholding> admm(splittingProblem);

            LASSOProblem<real_t> lassoProb(wlsProb, regTerm);
            FISTA<real_t> fista(lassoProb);

            THEN("the solutions match") { REQUIRE(isApprox(admm.solve(200), fista.solve(200))); }
        }

        WHEN("setting up ADMM to solve a WLSProblem + L0PseudoNorm")
        {
            L0PseudoNorm<real_t> regFunc(volDescr);
            RegularizationTerm<real_t> regTerm(0.000001f, regFunc);

            SplittingProblem<real_t> splittingProblem(wlsProb.getDataTerm(), regTerm, constraint);

            ADMM<CG, HardThresholding> admm(splittingProblem);

            THEN("the solution doesn't throw, is not nan and is approximate to the b vector")
            {
                REQUIRE_NOTHROW(admm.solve(200));
                REQUIRE(!std::isnan(admm.solve(200).squaredL2Norm()));
                REQUIRE(isApprox(admm.solve(200), dcB));
            }
        }
    }
}
