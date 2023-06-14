/**
 * @file test_CGNL.cpp
 *
 * @brief Tests for the CG class
 * @author Nikola Dinev - initial code
 * @author Eddie Groh - refactor and modifications for CGNL
 */

#include "doctest/doctest.h"

#include "CGNL.h"
#include "Scaling.h"
#include "Logger.h"
#include "VolumeDescriptor.h"
#include "JosephsMethod.h"
#include "CircleTrajectoryGenerator.h"
#include "Phantoms.h"
#include "TypeCasts.hpp"
#include "testHelpers.h"
#include "Quadric.h"
#include "WeightedLeastSquares.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("solvers");

template <template <typename> typename T, typename data_t>
constexpr data_t return_data_t(const T<data_t>&);

TYPE_TO_STRING(CGNL<float>);
TYPE_TO_STRING(CGNL<double>);

TEST_CASE_TEMPLATE("CGNL: Solving a simple linear problem", TestType, CGNL<float>)
{
    using data_t = decltype(return_data_t(std::declval<TestType>()));
    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("a linear problem")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 13, 24;
        VolumeDescriptor dd{numCoeff};

        Eigen::Matrix<data_t, -1, 1> bVec(dd.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer<data_t> dcB{dd, bVec};

        bVec.setRandom();
        bVec = bVec.cwiseAbs();
        Scaling<data_t> scalingOp{dd, DataContainer<data_t>{dd, bVec}};

        Quadric<data_t> prob{scalingOp, dcB};

        data_t epsilon = std::numeric_limits<data_t>::epsilon();

        WHEN("setting up a CGNL solver")
        {
            TestType solver{prob, epsilon};

            THEN("the clone works correctly")
            {
                auto cgnClone = solver.clone();

                REQUIRE_NE(cgnClone.get(), &solver);
                REQUIRE_EQ(*cgnClone, solver);

                AND_THEN("it works as expected")
                {
                    auto solution = solver.solve(dd.getNumberOfCoefficients() * 4);

                    DataContainer<data_t> resultsDifference = scalingOp.apply(solution) - dcB;

                    // should have converged for the given number of iterations
                    REQUIRE_LE((resultsDifference).squaredL2Norm(), dcB.squaredL2Norm());
                }
            }
        }
    }
}

TEST_CASE_TEMPLATE("CGNL: Solving a Tikhonov problem", TestType, CGNL<float>, CGNL<double>)
{
    using data_t = decltype(return_data_t(std::declval<TestType>()));
    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("a Tikhonov problem")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 13, 24;
        VolumeDescriptor dd{numCoeff};

        Eigen::Matrix<data_t, -1, 1> bVec(dd.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer<data_t> dcB{dd, bVec};

        bVec.setRandom();
        bVec = bVec.cwiseProduct(bVec);
        Scaling<data_t> scalingOp{dd, DataContainer<data_t>{dd, bVec}};

        auto lambda = static_cast<data_t>(0.1);
        Scaling<data_t> lambdaOp{dd, lambda};

        Quadric<data_t> prob{scalingOp + lambdaOp, dcB};

        data_t epsilon = std::numeric_limits<data_t>::epsilon();

        WHEN("setting up a CGNL solver")
        {
            TestType solver{prob, epsilon};

            THEN("the clone works correctly")
            {
                auto cgnClone = solver.clone();

                REQUIRE_NE(cgnClone.get(), &solver);
                REQUIRE_EQ(*cgnClone, solver);

                AND_THEN("it works as expected")
                {
                    auto solution = solver.solve(dd.getNumberOfCoefficients() * 4);

                    DataContainer<data_t> resultsDifference =
                        (scalingOp + lambdaOp).apply(solution) - dcB;

                    // should have converged for the given number of iterations
                    REQUIRE_LE(resultsDifference.squaredL2Norm(), dcB.squaredL2Norm());
                }
            }
        }
    }
}

TEST_CASE_TEMPLATE("CGNL: Solving a simple phantom reconstruction", TestType, CGNL<float>)
{
    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("a Phantom reconstruction problem")
    {

        IndexVector_t size(2);
        size << 16, 16; // TODO: determine optimal phantom size for efficient testing
        auto phantom = phantoms::modifiedSheppLogan(size);
        auto& volumeDescriptor = phantom.getDataDescriptor();

        index_t numAngles{30}, arc{180};
        auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(
            numAngles, phantom.getDataDescriptor(), arc, static_cast<real_t>(size(0)) * 100.0f,
            static_cast<real_t>(size(0)));

        JosephsMethod projector(downcast<VolumeDescriptor>(volumeDescriptor), *sinoDescriptor);

        auto sinogram = projector.apply(phantom);
        Quadric<real_t> problem{projector, sinogram};

        WHEN("setting up a CGNL solver")
        {
            TestType solver{problem};

            THEN("the clone works correctly")
            {
                auto cgnClone = solver.clone();

                REQUIRE_NE(cgnClone.get(), &solver);
                REQUIRE_EQ(*cgnClone, solver);

                AND_THEN("it works as expected")
                {
                    auto reconstruction = solver.solve(40);

                    DataContainer resultsDifference = reconstruction - phantom;

                    // should have converged for the given number of iterations
                    // does not converge to the optimal solution because of the regularization term
                    REQUIRE_UNARY(checkApproxEq(resultsDifference.squaredL2Norm(),
                                                phantom.squaredL2Norm(), 0.1));
                }
            }
        }
    }
}

TEST_SUITE_END();
