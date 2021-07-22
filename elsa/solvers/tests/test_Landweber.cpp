/**
 * \file test_Landweber.cpp
 *
 * \brief Tests for the Landweber class
 *
 * \author Maryna Shcherbak - initial code
 */

#include "doctest/doctest.h"
#include "Landweber.h"
#include "WLSProblem.h"
#include "Problem.h"
#include "Identity.h"
#include "LinearResidual.h"
#include "L2NormPow2.h"
#include "Logger.h"
#include "VolumeDescriptor.h"
#include "testHelpers.h"
#include <iostream>

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("solvers");

TYPE_TO_STRING(Landweber<float>);
TYPE_TO_STRING(Landweber<double>);

template <template <typename> typename T, typename data_t>
constexpr data_t return_data_t(const T<data_t>&);

TEST_CASE_TEMPLATE("Scenario: Solving a simple linear problem", TestType, Landweber<float>,
                   Landweber<double>)
{
    // Set seed for Eigen Matrices!
    srand((unsigned int) 666);

    using data_t = decltype(return_data_t(std::declval<TestType>()));
    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::WARN);

    GIVEN("a linear problem")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 13, 24;
        VolumeDescriptor dd(numCoeff);

        Eigen::Matrix<data_t, -1, 1> bVec(dd.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer dcB(dd, bVec);

        Identity<data_t> idOp(dd);

        WLSProblem prob(idOp, dcB);

        WHEN("setting up a Landweber solver with fixed step size")
        {
            TestType solver{prob, 0.5, TestType::Projected::NO};

            THEN("the clone works correctly")
            {
                auto lClone = solver.clone();

                REQUIRE_NE(lClone.get(), &solver);
                REQUIRE_EQ(*lClone, solver);

                AND_THEN("it works as expected")
                {
                    auto solution = solver.solve(1000);
                    REQUIRE_UNARY(checkApproxEq(solution.squaredL2Norm(), bVec.squaredNorm()));
                }
            }
        }

        WHEN("setting up a Landweber solver with lipschitz step size")
        {
            WLSProblem prob1(idOp, dcB);
            TestType solver{prob1};

            THEN("the clone works correctly")
            {
                auto lClone = solver.clone();

                REQUIRE_NE(lClone.get(), &solver);
                REQUIRE_EQ(*lClone, solver);

                AND_THEN("it works as expected")
                {
                    auto solution = solver.solve(1000);
                    REQUIRE_UNARY(checkApproxEq(solution.squaredL2Norm(), bVec.squaredNorm()));
                }
            }
        }

        WHEN("setting up a projected Landweber solver with fixed step size")
        {
            bVec = bVec.cwiseAbs();
            DataContainer dcBp(dd, bVec);
            dcB = dcBp;

            WLSProblem prob2(idOp, dcB);

            TestType solver{prob2, 0.5, TestType::Projected::YES};

            THEN("the clone works correctly")
            {
                auto lClone = solver.clone();

                REQUIRE_NE(lClone.get(), &solver);
                REQUIRE_EQ(*lClone, solver);

                AND_THEN("it works as expected")
                {
                    auto solution = solver.solve(1000);
                    REQUIRE_UNARY(checkApproxEq(solution.squaredL2Norm(), bVec.squaredNorm()));
                }
            }
        }

        WHEN("setting up a projected Landweber solver with lipschitz step size")
        {
            bVec = bVec.cwiseAbs(); //??
            DataContainer dcBp(dd, bVec);
            dcB = dcBp;

            WLSProblem prob2(idOp, dcB);

            TestType solver{prob2, TestType::Projected::YES};

            THEN("the clone works correctly")
            {
                auto lClone = solver.clone();

                REQUIRE_NE(lClone.get(), &solver);
                REQUIRE_EQ(*lClone, solver);

                AND_THEN("it works as expected")
                {
                    auto solution = solver.solve(1000);
                    REQUIRE_UNARY(checkApproxEq(solution.squaredL2Norm(), bVec.squaredNorm()));
                }
            }
        }
    }
}
