/**
 * @file test_GradientDescent.cpp
 *
 * @brief Tests for the GradientDescent class
 *
 * @author Tobias Lasser - initial code
 */

#include "doctest/doctest.h"

#include "GradientDescent.h"
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

TYPE_TO_STRING(GradientDescent<float>);
TYPE_TO_STRING(GradientDescent<double>);

template <template <typename> typename T, typename data_t>
constexpr data_t return_data_t(const T<data_t>&);

TEST_CASE_TEMPLATE("GradientDescent: Solving a simple linear problem", TestType,
                   GradientDescent<float>, GradientDescent<double>)
{
    // Set seed for Eigen Matrices!
    srand((unsigned int) 666);

    using data_t = decltype(return_data_t(std::declval<TestType>()));
    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("a linear problem")
    {
        VolumeDescriptor dd({13, 24});

        Eigen::Matrix<data_t, -1, 1> bVec(dd.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer dcB(dd, bVec);

        Identity<data_t> idOp(dd);

        WLSProblem prob(idOp, dcB);
        data_t epsilon = std::numeric_limits<data_t>::epsilon();

        WHEN("setting up a Gradient Descent solver with fixed step size")
        {
            TestType solver{prob, 0.5};

            THEN("the clone works correctly")
            {
                auto gdClone = solver.clone();

                REQUIRE_NE(gdClone.get(), &solver);
                REQUIRE_EQ(*gdClone, solver);

                AND_THEN("it works as expected")
                {
                    auto solution = solver.solve(100);

                    DataContainer<data_t> resultsDifference = solution - dcB;

                    // should have converged for the given number of iterations
                    REQUIRE_LE(resultsDifference.squaredL2Norm(),
                               epsilon * epsilon * dcB.squaredL2Norm());
                }
            }
        }

        WHEN("setting up a Gradient Descent solver with lipschitz step size")
        {
            TestType solver{prob};

            THEN("the clone works correctly")
            {
                auto gdClone = solver.clone();

                REQUIRE_NE(gdClone.get(), &solver);
                REQUIRE_EQ(*gdClone, solver);

                AND_THEN("it works as expected")
                {
                    auto solution = solver.solve(10);

                    DataContainer<data_t> resultsDifference = solution - dcB;

                    // should have converged for the given number of iterations
                    REQUIRE_UNARY(checkApproxEq(resultsDifference.squaredL2Norm(),
                                                epsilon * epsilon * dcB.squaredL2Norm()));
                }
            }
        }
    }
}

TEST_CASE_TEMPLATE("GradientDescent: Solving a Tikhonov problem", TestType, GradientDescent<float>,
                   GradientDescent<double>)
{
    // Set seed for Eigen Matrices!
    srand((unsigned int) 666);

    using data_t = decltype(return_data_t(std::declval<TestType>()));
    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("a Tikhonov problem")
    {
        VolumeDescriptor dd({13, 24});

        Eigen::Matrix<data_t, -1, 1> bVec(dd.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer dcB(dd, bVec);

        Identity<data_t> idOp(dd);
        LinearResidual<data_t> linRes(idOp, dcB);
        L2NormPow2<data_t> func(linRes);

        // the regularization term
        L2NormPow2<data_t> regFunc(dd);
        auto lambda = static_cast<data_t>(0.1f);
        RegularizationTerm<data_t> regTerm(lambda, regFunc);
        Problem prob(func, regTerm);

        WHEN("setting up a Gradient Descent solver with fixed step size")
        {
            TestType solver{prob, 0.5};

            THEN("the clone works correctly")
            {
                auto gdClone = solver.clone();

                REQUIRE_NE(gdClone.get(), &solver);
                REQUIRE_EQ(*gdClone, solver);
            }
            THEN("it works as expected")
            {
                auto solution = solver.solve(10);

                DataContainer<data_t> resultsDifference = solution - dcB;

                // should have converged for the given number of iterations
                // does not converge to the optimal solution because of the regularization term
                // Therefore, just check to fixed value
                REQUIRE_UNARY(checkApproxEq(resultsDifference.squaredL2Norm(), 0.85f));
            }
        }
    }
}

TEST_SUITE_END();
