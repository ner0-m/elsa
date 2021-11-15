/**
 * @file test_FGM.cpp
 *
 * @brief Tests for the Fast Gradient Method class
 *
 * @author Michael Loipführer - initial code
 */

#include "doctest/doctest.h"

#include <iostream>
#include "FGM.h"
#include "WLSProblem.h"
#include "Identity.h"
#include "Logger.h"
#include "VolumeDescriptor.h"
#include "SiddonsMethod.h"
#include "CircleTrajectoryGenerator.h"
#include "PhantomGenerator.h"
#include "JacobiPreconditioner.h"
#include "TypeCasts.hpp"
#include "testHelpers.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("solvers");

template <template <typename> typename T, typename data_t>
constexpr data_t return_data_t(const T<data_t>&);

TYPE_TO_STRING(FGM<float>);
TYPE_TO_STRING(FGM<double>);

TEST_CASE_TEMPLATE("FGM: Solving a simple linear problem", TestType, FGM<float>, FGM<double>)
{
    // Set seed for Eigen Matrices!
    srand((unsigned int) 666);

    using data_t = decltype(return_data_t(std::declval<TestType>()));
    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("a linear problem")
    {
        VolumeDescriptor dd{{13, 24}};

        Eigen::Matrix<data_t, -1, 1> bVec(dd.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer<data_t> dcB{dd, bVec};

        bVec.setRandom();
        bVec = bVec.cwiseAbs();
        Scaling<data_t> scalingOp{dd, DataContainer<data_t>{dd, bVec}};

        // using WLS problem here for ease of use
        WLSProblem prob{scalingOp, dcB};

        data_t epsilon = std::numeric_limits<data_t>::epsilon();

        WHEN("setting up a FGM solver")
        {
            TestType solver{prob, epsilon};

            THEN("the clone works correctly")
            {
                auto fgmClone = solver.clone();

                REQUIRE_NE(fgmClone.get(), &solver);
                REQUIRE_EQ(*fgmClone, solver);

                AND_THEN("it works as expected")
                {
                    auto solution = solver.solve(1000);

                    DataContainer<data_t> resultsDifference = scalingOp.apply(solution) - dcB;

                    // should have converged for the given number of iterations
                    REQUIRE_UNARY(checkApproxEq(resultsDifference.squaredL2Norm(),
                                                epsilon * epsilon * dcB.squaredL2Norm(), 0.5f));
                }
            }
        }

        WHEN("setting up a preconditioned FGM solver")
        {
            auto preconditionerInverse = JacobiPreconditioner<data_t>(scalingOp, true);
            TestType solver{prob, preconditionerInverse, epsilon};

            THEN("the clone works correctly")
            {
                auto fgmClone = solver.clone();

                REQUIRE_NE(fgmClone.get(), &solver);
                REQUIRE_EQ(*fgmClone, solver);

                AND_THEN("it works as expected")
                {
                    // with a good preconditioner we should need fewer iterations than without
                    auto solution = solver.solve(1000);

                    DataContainer<data_t> resultsDifference = scalingOp.apply(solution) - dcB;

                    // should have converged for the given number of iterations
                    REQUIRE_UNARY(checkApproxEq(resultsDifference.squaredL2Norm(),
                                                epsilon * epsilon * dcB.squaredL2Norm(), 0.1f));
                }
            }
        }
    }
}

TEST_CASE_TEMPLATE("FGM: Solving a Tikhonov problem", TestType, FGM<float>, FGM<double>)
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

        // the regularization term

        bVec.setRandom();
        bVec = bVec.cwiseProduct(bVec);
        Scaling<data_t> scalingOp{dd, DataContainer<data_t>{dd, bVec}};

        auto lambda = static_cast<data_t>(0.1);
        Scaling<data_t> lambdaOp{dd, lambda};

        // using WLS problem here for ease of use
        WLSProblem<data_t> prob{scalingOp + lambdaOp, dcB};

        data_t epsilon = std::numeric_limits<data_t>::epsilon();

        WHEN("setting up a FGM solver")
        {
            TestType solver{prob, epsilon};

            THEN("the clone works correctly")
            {
                auto fgmClone = solver.clone();

                REQUIRE_NE(fgmClone.get(), &solver);
                REQUIRE_EQ(*fgmClone, solver);

                AND_THEN("it works as expected")
                {
                    auto solution = solver.solve(dd.getNumberOfCoefficients());

                    DataContainer<data_t> resultsDifference =
                        (scalingOp + lambdaOp).apply(solution) - dcB;

                    // should have converged for the given number of iterations
                    // does not converge to the optimal solution because of the regularization term
                    REQUIRE_UNARY(checkApproxEq(resultsDifference.squaredL2Norm(),
                                                epsilon * epsilon * dcB.squaredL2Norm(), 0.1f));
                }
            }
        }

        WHEN("setting up a preconditioned FGM solver")
        {
            auto preconditionerInverse = JacobiPreconditioner<data_t>(scalingOp + lambdaOp, true);
            TestType solver{prob, preconditionerInverse, epsilon};

            THEN("the clone works correctly")
            {
                auto fgmClone = solver.clone();

                REQUIRE_NE(fgmClone.get(), &solver);
                REQUIRE_EQ(*fgmClone, solver);

                AND_THEN("it works as expected")
                {
                    // a perfect preconditioner should allow for convergence in a single step
                    auto solution = solver.solve(dd.getNumberOfCoefficients());

                    DataContainer<data_t> resultsDifference =
                        (scalingOp + lambdaOp).apply(solution) - dcB;

                    // should have converged for the given number of iterations
                    REQUIRE_UNARY(checkApproxEq(resultsDifference.squaredL2Norm(),
                                                epsilon * epsilon * dcB.squaredL2Norm(), 0.1f));
                }
            }
        }
    }
}

TEST_CASE("FGM: Solving a simple phantom reconstruction")
{
    // Set seed for Eigen Matrices!
    srand((unsigned int) 666);

    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("a Phantom reconstruction problem")
    {

        IndexVector_t size({{16, 16}});
        auto phantom = PhantomGenerator<real_t>::createModifiedSheppLogan(size);
        auto& volumeDescriptor = phantom.getDataDescriptor();

        index_t numAngles{30}, arc{180};
        auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(
            numAngles, phantom.getDataDescriptor(), arc, static_cast<real_t>(size(0)) * 100.0f,
            static_cast<real_t>(size(0)));

        SiddonsMethod projector(downcast<VolumeDescriptor>(volumeDescriptor), *sinoDescriptor);

        auto sinogram = projector.apply(phantom);

        WLSProblem problem(projector, sinogram);
        real_t epsilon = std::numeric_limits<real_t>::epsilon();

        WHEN("setting up a FGM solver")
        {
            FGM solver{problem, epsilon};

            THEN("the clone works correctly")
            {
                auto fgmClone = solver.clone();

                REQUIRE_NE(fgmClone.get(), &solver);
                REQUIRE_EQ(*fgmClone, solver);

                AND_THEN("it works as expected")
                {
                    auto reconstruction = solver.solve(15);

                    DataContainer resultsDifference = reconstruction - phantom;

                    // should have converged for the given number of iterations
                    // does not converge to the optimal solution because of the regularization term
                    REQUIRE(checkApproxEq(resultsDifference.squaredL2Norm(),
                                          epsilon * epsilon * phantom.squaredL2Norm(), 0.1));
                }
            }
        }
    }
}

TEST_SUITE_END();
