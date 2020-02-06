/**
 * \file test_GradientDescent.cpp
 *
 * \brief Tests for the GradientDescent class
 *
 * \author Tobias Lasser - initial code
 */

#include <catch2/catch.hpp>
#include "GradientDescent.h"
#include "WLSProblem.h"
#include "Problem.h"
#include "Identity.h"
#include "LinearResidual.h"
#include "L2NormPow2.h"
#include "Logger.h"

using namespace elsa;

SCENARIO("Solving a simple linear problem")
{
    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::WARN);

    GIVEN("a linear problem")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 13, 24;
        DataDescriptor dd(numCoeff);

        RealVector_t bVec(dd.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer dcB(dd, bVec);

        Identity idOp(dd);

        WLSProblem prob(idOp, dcB);

        WHEN("setting up a gd solver")
        {
            GradientDescent solver(prob, 0.1f);

            THEN("the clone works correctly")
            {
                auto gdClone = solver.clone();

                REQUIRE(gdClone.get() != &solver);
                REQUIRE(*gdClone == solver);
            }

            THEN("it works as expected")
            {
                auto solution = solver.solve(1000);
                REQUIRE(solution.squaredL2Norm() == Approx(bVec.squaredNorm()));
            }
        }
    }
}

SCENARIO("Solving a Tikhonov optimization problem")
{
    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::WARN);

    GIVEN("a Tikhonov problem")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 13, 24;
        DataDescriptor dd(numCoeff);

        RealVector_t bVec(dd.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer dcB(dd, bVec);

        Identity idOp(dd);
        LinearResidual linRes(idOp, dcB);
        L2NormPow2 func(linRes);

        // the regularization term
        L2NormPow2 regFunc(dd);
        RegularizationTerm regTerm(1.0f, regFunc);

        Problem prob(func, regTerm);

        WHEN("setting up a gd solver")
        {
            GradientDescent solver(prob, 0.5);

            THEN("the clone works correctly")
            {
                auto gdClone = solver.clone();

                REQUIRE(gdClone.get() != &solver);
                REQUIRE(*gdClone == solver);
            }

            THEN("it works as expected")
            {
                auto solution = solver.solve(1000);
                REQUIRE(solution.squaredL2Norm() == Approx(0.25 * bVec.squaredNorm()));
            }
        }
    }
}