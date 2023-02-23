/**
 * @file test_GradientDescent.cpp
 *
 * @brief Tests for the GradientDescent class
 *
 * @author Tobias Lasser - initial code
 */

#include "doctest/doctest.h"

#include "GradientDescent.h"
#include "Identity.h"
#include "LinearResidual.h"
#include "LeastSquares.h"
#include "L2Squared.h"
#include "Logger.h"
#include "VolumeDescriptor.h"
#include "testHelpers.h"
#include <iostream>

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("solvers");

TYPE_TO_STRING(GradientDescent<float>);
TYPE_TO_STRING(GradientDescent<double>);

template <template <typename> typename T, typename data_t>
constexpr data_t return_data_t(const T<data_t>&);

TEST_CASE_TEMPLATE("GradientDescent: Solving a simple linear problem", data_t, float, double)
{
    // Set seed for Eigen Matrices!
    srand((unsigned int) 666);

    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("a linear problem")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 13, 24;
        VolumeDescriptor dd(numCoeff);

        Eigen::Matrix<data_t, -1, 1> bVec(dd.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer dcB(dd, bVec);

        Identity<data_t> idOp(dd);

        LeastSquares<data_t> prob(idOp, dcB);

        WHEN("setting up a Gradient Descent solver with fixed step size")
        {
            GradientDescent<data_t> solver{prob, 0.5};

            THEN("the clone works correctly")
            {
                auto gdClone = solver.clone();

                REQUIRE_NE(gdClone.get(), &solver);
                REQUIRE_EQ(*gdClone, solver);

                AND_THEN("it works as expected")
                {
                    auto solution = solver.solve(100);

                    DataContainer<data_t> diff = solution - dcB;

                    // should have converged for the given number of iterations
                    CHECK_EQ(diff.squaredL2Norm(), doctest::Approx(0));
                }
            }
        }

        WHEN("setting up a Gradient Descent solver with lipschitz step size")
        {
            GradientDescent<data_t> solver{prob};

            THEN("the clone works correctly")
            {
                auto gdClone = solver.clone();

                REQUIRE_NE(gdClone.get(), &solver);
                REQUIRE_EQ(*gdClone, solver);

                AND_THEN("it works as expected")
                {
                    auto solution = solver.solve(50);

                    DataContainer<data_t> diff = solution - dcB;

                    // should have converged for the given number of iterations
                    CHECK_EQ(diff.squaredL2Norm(), doctest::Approx(0));
                }
            }
        }
    }
}

TEST_CASE_TEMPLATE("GradientDescent: Solving a Tikhonov problem", data_t, float, double)
{
    // Set seed for Eigen Matrices!
    srand((unsigned int) 666);

    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("a Tikhonov problem")
    {
        VolumeDescriptor dd({13, 24});

        Vector_t<data_t> bVec(dd.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer dcB(dd, bVec);

        Identity<data_t> op(dd);
        LeastSquares<data_t> fn(op, dcB);

        // the regularization term
        L2Squared<data_t> l2(dd);
        auto prob = fn + data_t{0.0001} * l2;

        WHEN("setting up a Gradient Descent solver with fixed step size")
        {
            GradientDescent<data_t> solver{prob, 0.5};

            THEN("the clone works correctly")
            {
                auto gdClone = solver.clone();

                CHECK_NE(gdClone.get(), &solver);
                CHECK_EQ(*gdClone, solver);
            }
            THEN("it works as expected")
            {
                auto solution = solver.solve(50);

                DataContainer<data_t> diff = solution - dcB;

                // should have converged for the given number of iterations
                CHECK_EQ(diff.squaredL2Norm(), doctest::Approx(0));
            }
        }
    }
}

TEST_SUITE_END();
