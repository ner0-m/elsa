#include "doctest/doctest.h"

#include "ADMM.h"
#include "ProximalL1.h"
#include "ProximalL0.h"
#include "CGLS.h"
#include "Identity.h"
#include "APGD.h"
#include "Logger.h"
#include "VolumeDescriptor.h"
#include <testHelpers.h>

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("solvers");

TEST_CASE_TEMPLATE("ADMM: Solving problems", data_t, float, double)
{
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("some problems and a constraint")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 21, 11;
        VolumeDescriptor volDescr(numCoeff);

        Vector_t<data_t> bVec(volDescr.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer<data_t> dcB(volDescr, bVec);

        Identity<data_t> idOp(volDescr);
        Scaling<data_t> negativeIdOp(volDescr, -1);
        DataContainer<data_t> dCC(volDescr);
        dCC = 0;

        WLSProblem<data_t> wlsProb(idOp, dcB);

        Constraint<data_t> constraint(idOp, negativeIdOp, dCC);

        WHEN("setting up ADMM and APGD to solve a LASSOProblem")
        {
            L1Norm<data_t> regFunc(volDescr);
            RegularizationTerm<data_t> regTerm(0.000001f, regFunc);

            SplittingProblem<data_t> splittingProblem(wlsProb.getDataTerm(), regTerm, constraint);

            ADMM<CGLS, ProximalL1, data_t> admm(splittingProblem);

            LASSOProblem<data_t> lassoProb(wlsProb, regTerm);
            APGD<data_t> fista(lassoProb);

            THEN("the solutions match")
            {
                auto sol1 = admm.solve(10);
                auto sol2 = fista.solve(10);
                CAPTURE(sol1);
                CAPTURE(sol2);
                REQUIRE_UNARY(isApprox(sol1, sol2, 0.1));
            }
        }

        WHEN("setting up ADMM to solve a WLSProblem + L0PseudoNorm")
        {
            L0PseudoNorm<data_t> regFunc(volDescr);
            RegularizationTerm<data_t> regTerm(0.000001f, regFunc);

            SplittingProblem<data_t> splittingProblem(wlsProb.getDataTerm(), regTerm, constraint);

            ADMM<CGLS, ProximalL1, data_t> admm(splittingProblem);

            THEN("the solution doesn't throw, is not nan and is approximate to the b vector")
            {
                REQUIRE_NOTHROW(admm.solve(10));
                REQUIRE_UNARY(!std::isnan(admm.solve(20).squaredL2Norm()));

                auto soladmm = admm.solve(20);
                CAPTURE(soladmm);
                CAPTURE(dcB);
                REQUIRE_UNARY(isApprox(soladmm, dcB, 0.1));
            }
        }
    }
}

TEST_SUITE_END();
