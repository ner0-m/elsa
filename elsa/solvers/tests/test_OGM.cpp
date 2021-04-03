/**
 * \file test_OGM.cpp
 *
 * \brief Tests for the Optimized Gradient Method class
 *
 * \author Michael Loipführer - initial code
 */

#include <catch2/catch.hpp>
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
#include "PhantomGenerator.h"

using namespace elsa;

template <template <typename> typename T, typename data_t>
constexpr data_t return_data_t(const T<data_t>&);

TEMPLATE_TEST_CASE("Scenario: Solving a simple linear problem", "", OGM<float>, OGM<double>)
{
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
                    REQUIRE(Approx(resultsDifference.squaredL2Norm()).margin(0.01)
                            <= epsilon * epsilon * dcB.squaredL2Norm());
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("Scenario: Solving a Tikhonov problem", "", OGM<float>, OGM<double>)
{
    using data_t = decltype(return_data_t(std::declval<TestType>()));
    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::WARN);

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
                    auto solution = solver.solve(dd.getNumberOfCoefficients());

                    DataContainer<data_t> resultsDifference = solution - dcB;

                    // should have converged for the given number of iterations
                    // does not converge to the optimal solution because of the regularization term
                    REQUIRE(Approx(resultsDifference.squaredL2Norm()).margin(1)
                            <= epsilon * epsilon * dcB.squaredL2Norm());
                }
            }
        }
    }
}

TEST_CASE("Scenario: Solving a simple phantom reconstruction", "")
{
    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::INFO);

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
            OGM solver{problem, epsilon};

            THEN("the clone works correctly")
            {
                auto ogmClone = solver.clone();

                REQUIRE(ogmClone.get() != &solver);
                REQUIRE(*ogmClone == solver);

                AND_THEN("it works as expected")
                {
                    auto reconstruction = solver.solve(40);

                    DataContainer resultsDifference = reconstruction - phantom;

                    // should have converged for the given number of iterations
                    // does not converge to the optimal solution because of the regularization term
                    REQUIRE(Approx(resultsDifference.squaredL2Norm()).margin(1)
                            <= epsilon * epsilon * phantom.squaredL2Norm());
                }
            }
        }
    }
}
