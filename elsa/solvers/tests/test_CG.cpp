/**
 * \file test_CG.cpp
 *
 * \brief Tests for the CG class
 *
 * \author Nikola Dinev
 */
#include <catch2/catch.hpp>

#include "CG.h"
#include "Identity.h"
#include "Scaling.h"
#include "Logger.h"
#include "L2NormPow2.h"

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
        DataDescriptor dd{numCoeff};

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

                    // should have converged for the given number of iterations
                    REQUIRE((scalingOp.apply(solution) - dcB).squaredL2Norm()
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

                    // should have converged for the given number of iterations
                    REQUIRE((scalingOp.apply(solution) - dcB).squaredL2Norm()
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
        DataDescriptor dd{numCoeff};

        Eigen::Matrix<data_t, -1, 1> bVec(dd.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer<data_t> dcB{dd, bVec};

        bVec.setRandom();
        bVec = bVec.cwiseProduct(bVec);
        Scaling<data_t> scalingOp{dd, DataContainer<data_t>{dd, bVec}};

        data_t lambda = 0.1;
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

                    // should have converged for the given number of iterations
                    REQUIRE(((scalingOp + lambdaOp).apply(solution) - dcB).squaredL2Norm()
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

                    // should have converged for the given number of iterations
                    REQUIRE(((scalingOp + lambdaOp).apply(solution) - dcB).squaredL2Norm()
                            <= epsilon * epsilon * dcB.squaredL2Norm());
                }
            }
        }
    }
}