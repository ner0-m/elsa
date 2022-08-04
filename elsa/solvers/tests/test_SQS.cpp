/**
 * @file test_SQS.cpp
 *
 * @brief Tests for the SQS class
 *
 * @author Michael Loipführer - initial code
 */

#include "doctest/doctest.h"

#include <iostream>
#include "JacobiPreconditioner.h"
#include "SQS.h"
#include "WLSProblem.h"
#include "WLSSubsetProblem.h"
#include "SubsetSampler.h"
#include "PlanarDetectorDescriptor.h"
#include "Identity.h"
#include "Logger.h"
#include "VolumeDescriptor.h"
#include "CircleTrajectoryGenerator.h"
#include "SiddonsMethod.h"
#include "PhantomGenerator.h"
#include "TypeCasts.hpp"
#include "testHelpers.h"

using namespace elsa;

template <template <typename> typename T, typename data_t>
constexpr data_t return_data_t(const T<data_t>&);

TEST_SUITE_BEGIN("solvers");

TYPE_TO_STRING(SQS<float>);
TYPE_TO_STRING(SQS<double>);

TEST_CASE_TEMPLATE("SQS: Solving a simple linear problem", TestType, SQS<float>, SQS<double>)
{
    // Set seed for Eigen Matrices!
    srand((unsigned int) 666);

    using data_t = decltype(return_data_t(std::declval<TestType>()));
    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("a linear problem")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 8, 14;
        VolumeDescriptor dd{numCoeff};

        Eigen::Matrix<data_t, -1, 1> bVec(dd.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer<data_t> dcB{dd, bVec};

        bVec.setRandom();
        bVec = bVec.cwiseAbs();
        Scaling<data_t> scalingOp{dd, DataContainer<data_t>{dd, bVec}};

        // using WLS problem here for ease of use
        WLSProblem prob{scalingOp, dcB};

        data_t epsilon = std::numeric_limits<data_t>::epsilon();

        WHEN("setting up a SQS solver")
        {
            TestType solver{prob, true, epsilon};

            THEN("the clone works correctly")
            {
                auto sqsClone = solver.clone();

                REQUIRE_NE(sqsClone.get(), &solver);
                REQUIRE_EQ(*sqsClone, solver);

                AND_THEN("it works as expected")
                {
                    auto solution = solver.solve(300);

                    DataContainer<data_t> resultsDifference = scalingOp.apply(solution) - dcB;

                    // should have converged for the given number of iterations
                    REQUIRE_UNARY(checkApproxEq(resultsDifference.squaredL2Norm(),
                                                epsilon * epsilon * dcB.squaredL2Norm()));
                }
            }
        }

        WHEN("setting up a preconditioned SQS solver")
        {
            auto preconditioner = JacobiPreconditioner<data_t>(scalingOp, false);
            TestType solver{prob, preconditioner, true, epsilon};

            THEN("the clone works correctly")
            {
                auto sqsClone = solver.clone();

                REQUIRE_NE(sqsClone.get(), &solver);
                REQUIRE_EQ(*sqsClone, solver);

                AND_THEN("it works as expected")
                {
                    // with a good preconditioner we should need fewer iterations than without
                    auto solution = solver.solve(200);

                    DataContainer<data_t> resultsDifference = scalingOp.apply(solution) - dcB;

                    // should have converged for the given number of iterations
                    REQUIRE_UNARY(checkApproxEq(resultsDifference.squaredL2Norm(),
                                                epsilon * epsilon * dcB.squaredL2Norm()));
                }
            }
        }
    }
}

TEST_CASE_TEMPLATE("SQS: Solving a Tikhonov problem", TestType, SQS<float>, SQS<double>)
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
        DataContainer dcB(dd, bVec);

        bVec.setRandom();
        bVec = bVec.cwiseProduct(bVec);
        Scaling<data_t> scalingOp{dd, DataContainer<data_t>{dd, bVec}};

        auto lambda = static_cast<data_t>(0.1);
        Scaling<data_t> lambdaOp{dd, lambda};

        // using WLS problem here for ease of use
        WLSProblem<data_t> prob{scalingOp + lambdaOp, dcB};

        data_t epsilon = std::numeric_limits<data_t>::epsilon();

        WHEN("setting up a SQS solver")
        {
            TestType solver{prob, true, epsilon};

            THEN("the clone works correctly")
            {
                auto sqsClone = solver.clone();

                REQUIRE_NE(sqsClone.get(), &solver);
                REQUIRE_EQ(*sqsClone, solver);

                AND_THEN("it works as expected")
                {
                    auto solution = solver.solve(dd.getNumberOfCoefficients());

                    DataContainer<data_t> resultsDifference =
                        (scalingOp + lambdaOp).apply(solution) - dcB;

                    // should have converged for the given number of iterations
                    // does not converge to the optimal solution because of the regularization term
                    REQUIRE_UNARY(checkApproxEq(resultsDifference.squaredL2Norm(),
                                                epsilon * epsilon * dcB.squaredL2Norm()));
                }
            }
        }

        WHEN("setting up a preconditioned SQS solver")
        {
            auto preconditioner = JacobiPreconditioner<data_t>(scalingOp + lambdaOp, false);
            TestType solver{prob, preconditioner, true, epsilon};

            THEN("the clone works correctly")
            {
                auto sqsClone = solver.clone();

                REQUIRE_NE(sqsClone.get(), &solver);
                REQUIRE_EQ(*sqsClone, solver);

                AND_THEN("it works as expected")
                {
                    // a perfect preconditioner should allow for convergence in a single step
                    auto solution = solver.solve(dd.getNumberOfCoefficients());

                    DataContainer<data_t> resultsDifference =
                        (scalingOp + lambdaOp).apply(solution) - dcB;

                    // should have converged for the given number of iterations
                    REQUIRE_UNARY(checkApproxEq(resultsDifference.squaredL2Norm(),
                                                epsilon * epsilon * dcB.squaredL2Norm()));
                }
            }
        }
    }
}

TEST_CASE("SQS: Solving a simple phantom reconstruction")
{
    // Set seed for Eigen Matrices!
    srand((unsigned int) 666);

    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("a Phantom reconstruction problem")
    {
        IndexVector_t size(2);
        size << 16, 16; // TODO: determine optimal phantom size for efficient testing
        auto phantom = PhantomGenerator<real_t>::createModifiedSheppLogan(size);
        auto& volumeDescriptor = phantom.getDataDescriptor();

        index_t numAngles{90}, arc{360};
        auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(
            numAngles, phantom.getDataDescriptor(), arc, static_cast<real_t>(size(0)) * 100.0f,
            static_cast<real_t>(size(0)));

        SiddonsMethod projector(downcast<VolumeDescriptor>(volumeDescriptor), *sinoDescriptor);

        auto sinogram = projector.apply(phantom);

        WLSProblem problem(projector, sinogram);
        real_t epsilon = std::numeric_limits<real_t>::epsilon();

        WHEN("setting up a SQS solver")
        {
            SQS solver{problem, true, epsilon};

            THEN("the clone works correctly")
            {
                auto sqsClone = solver.clone();

                REQUIRE_NE(sqsClone.get(), &solver);
                REQUIRE_EQ(*sqsClone, solver);

                AND_THEN("it works as expected")
                {
                    auto reconstruction = solver.solve(40);

                    DataContainer resultsDifference = reconstruction - phantom;

                    // should have converged for the given number of iterations
                    // does not converge to the optimal solution because of the regularization term
                    REQUIRE_UNARY(checkApproxEq(resultsDifference.squaredL2Norm(), 0.034f));
                }
            }
        }
    }
}

TEST_CASE("SQS: Solving a simple phantom problem using ordered subsets")
{
    // Set seed for Eigen Matrices!
    srand((unsigned int) 666);

    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("a Phantom reconstruction problem")
    {

        IndexVector_t size(2);
        size << 16, 16; // TODO: determine optimal phantom size for efficient testing
        auto phantom = PhantomGenerator<real_t>::createModifiedSheppLogan(size);
        auto& volumeDescriptor = phantom.getDataDescriptor();

        index_t numAngles{20}, arc{180};
        auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(
            numAngles, phantom.getDataDescriptor(), arc, static_cast<real_t>(size(0)) * 100.0f,
            static_cast<real_t>(size(0)));

        SiddonsMethod projector(static_cast<const VolumeDescriptor&>(volumeDescriptor),
                                *sinoDescriptor);

        auto sinogram = projector.apply(phantom);

        real_t epsilon = std::numeric_limits<real_t>::epsilon();

        WHEN("setting up a SQS solver with ROUND_ROBIN subsampling")
        {
            index_t nSubsets{4};
            SubsetSampler<PlanarDetectorDescriptor, real_t> subsetSampler(
                static_cast<const VolumeDescriptor&>(volumeDescriptor),
                static_cast<const PlanarDetectorDescriptor&>(*sinoDescriptor), nSubsets);

            WLSSubsetProblem<real_t> problem(
                *subsetSampler.getProjector<SiddonsMethod<real_t>>(),
                subsetSampler.getPartitionedData(sinogram),
                subsetSampler.getSubsetProjectors<SiddonsMethod<real_t>>());
            SQS solver{problem, true, epsilon};

            THEN("the clone works correctly")
            {
                auto sqsClone = solver.clone();

                REQUIRE(sqsClone.get() != &solver);
                REQUIRE(*sqsClone == solver);

                AND_THEN("it works as expected")
                {
                    auto reconstruction = solver.solve(10);

                    DataContainer resultsDifference = reconstruction - phantom;

                    // should have converged for the given number of iterations
                    // does not converge to the optimal solution because of the regularization term
                    REQUIRE_UNARY(checkApproxEq(resultsDifference.squaredL2Norm(),
                                                epsilon * epsilon * phantom.squaredL2Norm(), 0.1));
                }
            }
        }
        WHEN("setting up a SQS solver with ROTATIONAL_CLUSTERING subsampling")
        {
            index_t nSubsets{4};
            SubsetSampler<PlanarDetectorDescriptor, real_t> subsetSampler(
                static_cast<const VolumeDescriptor&>(volumeDescriptor),
                static_cast<const PlanarDetectorDescriptor&>(*sinoDescriptor), nSubsets,
                SubsetSampler<PlanarDetectorDescriptor,
                              real_t>::SamplingStrategy::ROTATIONAL_CLUSTERING);

            WLSSubsetProblem<real_t> problem(
                *subsetSampler.getProjector<SiddonsMethod<real_t>>(),
                subsetSampler.getPartitionedData(sinogram),
                subsetSampler.getSubsetProjectors<SiddonsMethod<real_t>>());
            SQS solver{problem, true, epsilon};

            THEN("the clone works correctly")
            {
                auto sqsClone = solver.clone();

                REQUIRE_NE(sqsClone.get(), &solver);
                REQUIRE_EQ(*sqsClone, solver);

                AND_THEN("it works as expected")
                {
                    auto reconstruction = solver.solve(10);

                    DataContainer resultsDifference = reconstruction - phantom;

                    // should have converged for the given number of iterations
                    REQUIRE_UNARY(checkApproxEq(resultsDifference.squaredL2Norm(),
                                                epsilon * epsilon * phantom.squaredL2Norm(), 0.1));
                }
            }
        }
    }
}

TEST_SUITE_END();
