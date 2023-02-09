#include "doctest/doctest.h"

#include "Error.h"
#include "APGD.h"
#include "Identity.h"
#include "Logger.h"
#include "VolumeDescriptor.h"
#include "QuadricProblem.h"
#include "testHelpers.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("solvers");

TEST_CASE("APGD: Solving a LASSOProblem")
{
    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("a LASSOProblem")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 25, 31;
        VolumeDescriptor volDescr(numCoeff);

        RealVector_t bVec(volDescr.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer dcB(volDescr, bVec);

        Identity idOp(volDescr);

        WLSProblem wlsProb(idOp, dcB);

        L1Norm regFunc(volDescr);
        RegularizationTerm regTerm(0.000001f, regFunc);

        LASSOProblem lassoProb(wlsProb, regTerm);

        WHEN("setting up a APGD solver")
        {
            APGD solver(lassoProb, geometry::Threshold(0.005f));

            THEN("cloned APGD solver equals original APGD solver")
            {
                auto fistaClone = solver.clone();

                REQUIRE_NE(fistaClone.get(), &solver);
                REQUIRE_EQ(*fistaClone, solver);
            }

            THEN("the solution is correct")
            {
                auto solution = solver.solve(500);
                REQUIRE_UNARY(checkApproxEq(solution.squaredL2Norm(), bVec.squaredNorm()));
            }
        }
    }
}

TEST_CASE("APGD: Solving various problems")
{
    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("a DataContainer")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 14, 9;
        VolumeDescriptor volDescr(numCoeff);

        RealVector_t bVec(volDescr.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer dcB(volDescr, bVec);

        WHEN("setting up an APGD solver for a WLSProblem")
        {
            Identity idOp(volDescr);

            WLSProblem wlsProb(idOp, dcB);

            THEN("an exception is thrown as no regularization term is provided")
            {
                REQUIRE_THROWS_AS(APGD{wlsProb}, InvalidArgumentError);
            }
        }

        WHEN("setting up an APGD solver for a QuadricProblem without A and without b")
        {
            Identity idOp(volDescr);

            QuadricProblem<real_t> quadricProbWithoutAb(Quadric<real_t>{volDescr});

            THEN("the vector b is initialized with zeroes and the operator A becomes an "
                 "identity operator but an exception is thrown due to missing regularization term")
            {
                REQUIRE_THROWS_AS(APGD{quadricProbWithoutAb}, InvalidArgumentError);
            }
        }
    }
}

TEST_SUITE_END();
