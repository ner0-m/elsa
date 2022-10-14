/**
 * @file test_OGM.cpp
 *
 * @brief Tests for the Optimized Gradient Method class
 *
 * @author Michael Loipführer - initial code
 */

#include "doctest/doctest.h"

#include <iostream>
#include "OGM.h"
#include "WLSProblem.h"
#include "Problem.h"
#include "Identity.h"
#include "LinearResidual.h"
#include "L2NormPow2.h"
#include "Logger.h"
#include "VolumeDescriptor.h"
#include "SiddonsMethod.h"
#include "CircleTrajectoryGenerator.h"
#include "Phantoms.h"
#include "TypeCasts.hpp"
#include "testHelpers.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("solvers");

TYPE_TO_STRING(OGM<float>);
TYPE_TO_STRING(OGM<double>);

template <template <typename> typename T, typename data_t>
constexpr data_t return_data_t(const T<data_t>&);

TEST_CASE_TEMPLATE("OGM: Solving a simple linear problem", TestType, OGM<float>, OGM<double>)
{
    // Set seed for Eigen Matrices!
    srand((unsigned int) 666);

    using data_t = decltype(return_data_t(std::declval<TestType>()));
    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("a linear problem")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 13, 15;
        VolumeDescriptor dd{numCoeff};

        Eigen::Matrix<data_t, -1, 1> bVec(dd.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer<data_t> b{dd, bVec};

        bVec.setRandom();
        bVec = bVec.cwiseAbs();
        Scaling<data_t> scalingOp{dd, DataContainer<data_t>{dd, bVec}};

        // using WLS problem here for ease of use
        // since OGM is very picky with the precision of the lipschitz constant of a problem we need
        // to pass it explicitly
        WLSProblem<data_t> prob{scalingOp, b, static_cast<data_t>(1.0)};

        data_t epsilon = std::numeric_limits<data_t>::epsilon();

        WHEN("setting up an OGM solver")
        {
            TestType solver{prob, epsilon};

            THEN("the clone works correctly")
            {
                auto ogmClone = solver.clone();

                REQUIRE_NE(ogmClone.get(), &solver);
                REQUIRE_EQ(*ogmClone, solver);

                AND_THEN("it works as expected")
                {
                    auto solution = solver.solve(100);

                    DataContainer<data_t> resultsDifference = scalingOp.apply(solution) - b;

                    // should have converged for the given number of iterations
                    REQUIRE_UNARY(checkApproxEq(resultsDifference.squaredL2Norm(),
                                                epsilon * epsilon * b.squaredL2Norm(), 0.5f));
                }
            }
        }

        WHEN("setting up a preconditioned OGM solver")
        {
            bVec = 1 / bVec.array();
            TestType solver{prob, Scaling<data_t>{dd, DataContainer<data_t>{dd, bVec}}, epsilon};

            THEN("the clone works correctly")
            {
                auto ogmClone = solver.clone();

                REQUIRE_NE(ogmClone.get(), &solver);
                REQUIRE_EQ(*ogmClone, solver);

                AND_THEN("it works as expected")
                {
                    // with a good preconditioner we should need fewer iterations than without
                    auto solution = solver.solve(100);

                    DataContainer<data_t> resultsDifference = scalingOp.apply(solution) - b;

                    // should have converged for the given number of iterations
                    REQUIRE_UNARY(checkApproxEq(resultsDifference.squaredL2Norm(),
                                                epsilon * epsilon * b.squaredL2Norm()));
                }
            }
        }
    }
}

TEST_CASE_TEMPLATE("OGM: Solving a Tikhonov problem", TestType, OGM<float>, OGM<double>)
{
    // Set seed for Eigen Matrices!
    srand((unsigned int) 666);

    using data_t = decltype(return_data_t(std::declval<TestType>()));
    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("a Tikhonov problem")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 13, 24;
        VolumeDescriptor dd(numCoeff);

        Eigen::Matrix<data_t, -1, 1> bVec(dd.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer b(dd, bVec);

        bVec.setRandom();
        bVec = bVec.cwiseProduct(bVec);
        Scaling<data_t> scalingOp{dd, DataContainer<data_t>{dd, bVec}};

        auto lambda = static_cast<data_t>(0.1);
        Scaling<data_t> lambdaOp{dd, lambda};

        // using WLS problem here for ease of use
        // since OGM is very picky with the precision of the lipschitz constant of a problem we need
        // to pass it explicitly
        WLSProblem<data_t> prob{scalingOp + lambdaOp, b, static_cast<data_t>(1.2)};

        data_t epsilon = std::numeric_limits<data_t>::epsilon();

        WHEN("setting up an OGM solver")
        {
            TestType solver{prob, epsilon};

            THEN("the clone works correctly")
            {
                auto ogmClone = solver.clone();

                REQUIRE_NE(ogmClone.get(), &solver);
                REQUIRE_EQ(*ogmClone, solver);

                AND_THEN("it works as expected")
                {
                    auto solution = solver.solve(dd.getNumberOfCoefficients());

                    DataContainer<data_t> resultsDifference =
                        (scalingOp + lambdaOp).apply(solution) - b;

                    // should have converged for the given number of iterations
                    // does not converge to the optimal solution because of the regularization term
                    REQUIRE_UNARY(checkApproxEq(resultsDifference.squaredL2Norm(),
                                                epsilon * epsilon * b.squaredL2Norm()));
                }
            }
        }

        WHEN("setting up a preconditioned OGM solver")
        {
            bVec = 1 / (bVec.array() + lambda);
            TestType solver{prob, Scaling<data_t>{dd, DataContainer<data_t>{dd, bVec}}, epsilon};

            THEN("the clone works correctly")
            {
                auto ogmClone = solver.clone();

                REQUIRE_NE(ogmClone.get(), &solver);
                REQUIRE_EQ(*ogmClone, solver);

                AND_THEN("it works as expected")
                {
                    // a perfect preconditioner should allow for convergence in a single step
                    auto solution = solver.solve(dd.getNumberOfCoefficients());

                    DataContainer<data_t> resultsDifference =
                        (scalingOp + lambdaOp).apply(solution) - b;

                    // should have converged for the given number of iterations
                    REQUIRE_UNARY(checkApproxEq(resultsDifference.squaredL2Norm(),
                                                epsilon * epsilon * b.squaredL2Norm()));
                }
            }
        }
    }
}

TEST_CASE("OGM: Solving a simple phantom reconstruction")
{
    // Set seed for Eigen Matrices!
    srand((unsigned int) 666);

    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("a Phantom reconstruction problem")
    {
        IndexVector_t size(2);
        size << 16, 16; // TODO: determine optimal phantom size for efficient testing
        auto phantom = phantoms::modifiedSheppLogan(size);
        auto& volumeDescriptor = phantom.getDataDescriptor();

        index_t numAngles{20}, arc{360};
        auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(
            numAngles, phantom.getDataDescriptor(), arc, static_cast<real_t>(size(0)) * 100.0f,
            static_cast<real_t>(size(0)));

        SiddonsMethod projector(downcast<VolumeDescriptor>(volumeDescriptor), *sinoDescriptor);

        auto sinogram = projector.apply(phantom);

        WLSProblem problem(projector, sinogram);
        real_t epsilon = std::numeric_limits<real_t>::epsilon();

        WHEN("setting up a SQS solver")
        {
            OGM solver{problem, epsilon};

            THEN("the clone works correctly")
            {
                auto ogmClone = solver.clone();

                REQUIRE_NE(ogmClone.get(), &solver);
                REQUIRE_EQ(*ogmClone, solver);

                AND_THEN("it works as expected")
                {
                    auto reconstruction = solver.solve(15);

                    DataContainer resultsDifference = reconstruction - phantom;

                    // should have converged for the given number of iterations
                    // does not converge to the optimal solution because of the regularization term
                    REQUIRE_UNARY(checkApproxEq(resultsDifference.squaredL2Norm(),
                                                epsilon * epsilon * phantom.squaredL2Norm(), 0.15));
                }
            }
        }
    }
}

TEST_SUITE_END();
