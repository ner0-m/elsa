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
#include "Problem.h"
#include "Identity.h"
#include "LinearResidual.h"
#include "L2NormPow2.h"
#include "Logger.h"
#include "VolumeDescriptor.h"
#include "SiddonsMethod.h"
#include "CircleTrajectoryGenerator.h"
#include "PhantomGenerator.h"
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
        IndexVector_t numCoeff(2);
        numCoeff << 13, 24;
        VolumeDescriptor dd(numCoeff);

        Eigen::Matrix<data_t, -1, 1> bVec(dd.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer dcB(dd, bVec);

        Identity<data_t> idOp(dd);

        WLSProblem prob(idOp, dcB);
        data_t epsilon = std::numeric_limits<data_t>::epsilon();

        WHEN("setting up a FGM solver")
        {
            TestType solver{prob, epsilon};

            THEN("the clone works correctly")
            {
                auto fgmClone = solver.clone();

                REQUIRE(fgmClone.get() != &solver);
                REQUIRE(*fgmClone == solver);

                AND_THEN("it works as expected")
                {
                    auto solution = solver.solve(dd.getNumberOfCoefficients());

                    DataContainer<data_t> resultsDifference = solution - dcB;

                    // should have converged for the given number of iterations
                    REQUIRE(resultsDifference.squaredL2Norm()
                            <= epsilon * epsilon * dcB.squaredL2Norm());
                    REQUIRE(checkApproxEq(resultsDifference.squaredL2Norm(),
                                          epsilon * epsilon * dcB.squaredL2Norm(), 0.1f));
                    REQUIRE(checkApproxEq(solution.squaredL2Norm(), dcB.squaredL2Norm(), 0.1f));
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
        IndexVector_t numCoeff(2);
        numCoeff << 13, 24;
        VolumeDescriptor dd(numCoeff);

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

        data_t epsilon = std::numeric_limits<data_t>::epsilon();

        WHEN("setting up a FGM solver")
        {
            TestType solver{prob, epsilon};

            THEN("the clone works correctly")
            {
                auto fgmClone = solver.clone();

                REQUIRE(fgmClone.get() != &solver);
                REQUIRE(*fgmClone == solver);

                AND_THEN("it works as expected")
                {
                    auto solution = fgmClone->solve(10);

                    DataContainer<data_t> resultsDifference = solution - dcB;

                    // should have converged for the given number of iterations
                    // does not converge to the optimal solution because of the regularization term
                    REQUIRE_UNARY(checkApproxEq(resultsDifference.squaredL2Norm(), 0.85));
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

        IndexVector_t size(2);
        size << 16, 16; // TODO: determine optimal phantom size for efficient testing
        auto phantom = PhantomGenerator<real_t>::createModifiedSheppLogan(size);
        auto& volumeDescriptor = phantom.getDataDescriptor();

        index_t numAngles{90}, arc{360};
        auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(
            numAngles, phantom.getDataDescriptor(), arc, static_cast<real_t>(size(0) * 100),
            static_cast<real_t>(size(0)));

        SiddonsMethod projector(dynamic_cast<const VolumeDescriptor&>(volumeDescriptor),
                                *sinoDescriptor);

        auto sinogram = projector.apply(phantom);

        WLSProblem problem(projector, sinogram);
        real_t epsilon = std::numeric_limits<real_t>::epsilon();

        WHEN("setting up a SQS solver")
        {
            FGM solver{problem, epsilon};

            THEN("the clone works correctly")
            {
                auto fgmClone = solver.clone();

                REQUIRE(fgmClone.get() != &solver);
                REQUIRE(*fgmClone == solver);

                AND_THEN("it works as expected")
                {
                    auto reconstruction = solver.solve(20);

                    DataContainer resultsDifference = reconstruction - phantom;

                    // should have converged for the given number of iterations
                    // does not converge to the optimal solution because of the regularization term
                    REQUIRE(checkApproxEq(resultsDifference.squaredL2Norm(),
                                          epsilon * epsilon * phantom.squaredL2Norm(), 0.1));
                    REQUIRE(checkApproxEq(reconstruction.squaredL2Norm(), phantom.squaredL2Norm()));
                }
            }
        }
    }
}

TEST_SUITE_END();
