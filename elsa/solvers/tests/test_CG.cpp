/**
 * @file test_CG.cpp
 *
 * @brief Tests for the CG class
 *
 * @author Nikola Dinev
 */
#include <catch2/catch.hpp>

#include "CG.h"
#include "Scaling.h"
#include "Logger.h"
#include "VolumeDescriptor.h"
#include "SiddonsMethod.h"
#include "CircleTrajectoryGenerator.h"
#include "PhantomGenerator.h"

using namespace elsa;

template <template <typename> typename T, typename data_t>
constexpr data_t return_data_t(const T<data_t>&);

TEMPLATE_TEST_CASE("Scenario: Solving a simple linear problem", "", CG<float>, CG<double>)
{
    using data_t = decltype(return_data_t(std::declval<TestType>()));
    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::WARN);

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

        QuadricProblem<data_t> prob{scalingOp, dcB, true};

        data_t epsilon = std::numeric_limits<data_t>::epsilon();

        WHEN("setting up a cg solver")
        {
            TestType solver{prob, epsilon};

            THEN("the clone works correctly")
            {
                auto cgClone = solver.clone();

                REQUIRE(cgClone.get() != &solver);
                REQUIRE(*cgClone == solver);

                AND_THEN("it works as expected")
                {
                    auto solution = solver.solve(dd.getNumberOfCoefficients());

                    DataContainer<data_t> resultsDifference = scalingOp.apply(solution) - dcB;

                    // should have converged for the given number of iterations
                    REQUIRE((resultsDifference).squaredL2Norm()
                            <= epsilon * epsilon * dcB.squaredL2Norm());
                }
            }
        }

        WHEN("setting up a preconditioned cg solver")
        {
            bVec = 1 / bVec.array();
            TestType solver{prob, Scaling<data_t>{dd, DataContainer<data_t>{dd, bVec}}, epsilon};

            THEN("the clone works correctly")
            {
                auto cgClone = solver.clone();

                REQUIRE(cgClone.get() != &solver);
                REQUIRE(*cgClone == solver);

                AND_THEN("it works as expected")
                {
                    // a perfect preconditioner should allow for convergence in a single step
                    auto solution = solver.solve(1);

                    DataContainer<data_t> resultsDifference = scalingOp.apply(solution) - dcB;

                    // should have converged for the given number of iterations
                    REQUIRE(Approx((resultsDifference).squaredL2Norm()).margin(0.00001)
                            <= epsilon * epsilon * dcB.squaredL2Norm());
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("Scenario: Solving a Tikhonov problem", "", CG<float>, CG<double>)
{
    using data_t = decltype(return_data_t(std::declval<TestType>()));
    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::WARN);

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

        QuadricProblem<data_t> prob{scalingOp + lambdaOp, dcB, true};

        data_t epsilon = std::numeric_limits<data_t>::epsilon();

        WHEN("setting up a cg solver")
        {
            TestType solver{prob, epsilon};

            THEN("the clone works correctly")
            {
                auto cgClone = solver.clone();

                REQUIRE(cgClone.get() != &solver);
                REQUIRE(*cgClone == solver);

                AND_THEN("it works as expected")
                {
                    auto solution = solver.solve(dd.getNumberOfCoefficients());

                    DataContainer<data_t> resultsDifference =
                        (scalingOp + lambdaOp).apply(solution) - dcB;

                    // should have converged for the given number of iterations
                    REQUIRE(resultsDifference.squaredL2Norm()
                            <= epsilon * epsilon * dcB.squaredL2Norm());
                }
            }
        }

        WHEN("setting up a preconditioned cg solver")
        {
            bVec = 1 / (bVec.array() + lambda);
            TestType solver{prob, Scaling<data_t>{dd, DataContainer<data_t>{dd, bVec}}, epsilon};

            THEN("the clone works correctly")
            {
                auto cgClone = solver.clone();

                REQUIRE(cgClone.get() != &solver);
                REQUIRE(*cgClone == solver);

                AND_THEN("it works as expected")
                {
                    // a perfect preconditioner should allow for convergence in a single step
                    auto solution = solver.solve(1);

                    DataContainer<data_t> resultsDifference =
                        (scalingOp + lambdaOp).apply(solution) - dcB;

                    // should have converged for the given number of iterations
                    REQUIRE(resultsDifference.squaredL2Norm()
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
            CG solver{problem, epsilon};

            THEN("the clone works correctly")
            {
                auto cgClone = solver.clone();

                REQUIRE(cgClone.get() != &solver);
                REQUIRE(*cgClone == solver);

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
