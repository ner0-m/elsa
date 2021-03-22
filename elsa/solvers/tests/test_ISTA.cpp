/**
 * @file test_ISTA.cpp
 *
 * @brief Tests for the ISTA class
 *
 * @author Andi Braimllari
 */

#include "ISTA.h"
#include "Identity.h"
#include "Logger.h"
#include "VolumeDescriptor.h"
#include "QuadricProblem.h"

#include <catch2/catch.hpp>

using namespace elsa;

SCENARIO("Solving a LASSOProblem with ISTA")
{
    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::WARN);

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

        WHEN("setting up an ISTA solver")
        {
            ISTA solver(lassoProb);

            THEN("cloned ISTA solver equals original ISTA solver")
            {
                auto istaClone = solver.clone();

                REQUIRE(istaClone.get() != &solver);
                REQUIRE(*istaClone == solver);
            }

            THEN("the solution is correct")
            {
                auto solution = solver.solve(1000);
                REQUIRE(solution.squaredL2Norm() == Approx(bVec.squaredNorm()));
            }
        }
    }
}

SCENARIO("Solving various problems with ISTA")
{
    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::WARN);

    GIVEN("a DataContainer")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 25, 31;
        VolumeDescriptor volDescr(numCoeff);

        RealVector_t bVec(volDescr.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer dcB(volDescr, bVec);

        WHEN("setting up an ISTA solver for a WLSProblem")
        {
            Identity idOp(volDescr);

            WLSProblem wlsProb(idOp, dcB);

            THEN("an exception is thrown as no regularization term is provided")
            {
                REQUIRE_THROWS_AS(ISTA(wlsProb), std::logic_error);
            }
        }

        WHEN("setting up an ISTA solver for a QuadricProblem without A and without b")
        {
            Identity idOp(volDescr);

            QuadricProblem<real_t> quadricProbWithoutAb(Quadric<real_t>{volDescr});

            THEN("the vector b is initialized with zeroes and the operator A becomes an "
                 "identity operator but an exception is thrown due to missing regularization term")
            {
                REQUIRE_THROWS_AS(ISTA(quadricProbWithoutAb), std::logic_error);
            }
        }
    }
}
