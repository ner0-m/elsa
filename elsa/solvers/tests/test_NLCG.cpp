
/**
 * @file test_NLCG.cpp
 *
 * @brief Tests for the NLCG class
 *
 * @author Maryna Shcherbak
 */

#include "doctest/doctest.h"
#include "NLCG.h"
#include "Identity.h"
#include "LinearResidual.h"
#include "Logger.h"
#include "VolumeDescriptor.h"
#include "testHelpers.h"
#include "Huber.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("solvers");

TYPE_TO_STRING(NLCG<float>);
TYPE_TO_STRING(NLCG<double>);

template <template <typename> typename T, typename data_t>
constexpr data_t return_data_t(const T<data_t>&);

TEST_CASE_TEMPLATE("Scenario: Solving a problem with a huber functional", TestType, NLCG<float>,
                   NLCG<double>)
{
    // Set seed for Eigen Matrices!
    srand((unsigned int) 666);

    using data_t = decltype(return_data_t(std::declval<TestType>()));
    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::OFF);
    using Vector = Eigen::Matrix<data_t, Eigen::Dynamic, 1>;

    GIVEN("a linear problem")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 47, 11;
        VolumeDescriptor dd{numCoeff};

        Vector bVec(dd.getNumberOfCoefficients());
        bVec.setRandom();
        bVec = bVec.cwiseAbs();

        DataContainer<data_t> dcB{dd, bVec};

        Vector h(dd.getNumberOfCoefficients());
        h.setRandom();
        // bVec = bVec.cwiseAbs();

        DataContainer<data_t> dch{dd, h};
        dch = 1;
        // Scaling<data_t> scalingOp{dd, DataContainer<data_t>{dd, bVec}};

        Identity<data_t> A(dd);

        LinearResidual<data_t> linRes(A, dch);

        // generisches Problem
        data_t epsilon = std::numeric_limits<data_t>::epsilon();
        //   Scaling<data_t> scalingOp{dd, DataContainer<data_t>{dd, bVec}};
        //    QuadricProblem<data_t> prob{scalingOp, dcB, true};

        Huber<data_t> func{linRes, 20};
        Problem<data_t> prob{func, dcB, true};
        WHEN("setting up a NLCG solver with Fletcher Reeves")
        {
            TestType solver{prob, TestType::Beta::FR};

            THEN("the clone works correctly")
            {
                auto nlcgClone = solver.clone();

                REQUIRE_NE(nlcgClone.get(), &solver);
                REQUIRE_EQ(*nlcgClone, solver);

                AND_THEN("it works as expected")
                {
                    auto solution = solver.solve(5);

                    // DataContainer<data_t> resultsDifference = scalingOp.apply(solution) - dcB;
                    DataContainer<data_t> resultsDifference = solution - dcB;
                    // should have converged for the given number of iterations
                    REQUIRE_LE((resultsDifference).squaredL2Norm(),
                               epsilon * epsilon * dcB.squaredL2Norm());
                }
            }
        }

        WHEN("setting up a NLCG solver with Polak Riviere")
        {
            TestType solver{prob, TestType::Beta::PR};

            THEN("the clone works correctly")
            {
                auto nlcgClone = solver.clone();

                REQUIRE_NE(nlcgClone.get(), &solver);
                REQUIRE_EQ(*nlcgClone, solver);

                AND_THEN("it works as expected")
                {
                    auto solution = solver.solve(dd.getNumberOfCoefficients());

                    // DataContainer<data_t> resultsDifference = scalingOp.apply(solution) - dcB;
                    DataContainer<data_t> resultsDifference = solution - dcB;
                    // should have converged for the given number of iterations
                    REQUIRE_LE((resultsDifference).squaredL2Norm(),
                               epsilon * epsilon * dcB.squaredL2Norm());
                }
            }
        }
    }
}

TEST_CASE_TEMPLATE("Scenario: Solving a quadric problem", TestType, NLCG<float>, NLCG<double>)
{
    // Set seed for Eigen Matrices!
    srand((unsigned int) 666);

    using data_t = decltype(return_data_t(std::declval<TestType>()));
    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::OFF);
    using Vector = Eigen::Matrix<data_t, Eigen::Dynamic, 1>;

    GIVEN("a linear problem")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 13, 24;
        VolumeDescriptor dd{numCoeff};

        Vector bVec(dd.getNumberOfCoefficients());
        bVec.setRandom();
        bVec = bVec.cwiseAbs();
        DataContainer<data_t> dcB{dd, bVec};

        // generisches Problem
        data_t epsilon = std::numeric_limits<data_t>::epsilon();
        Scaling<data_t> scalingOp{dd, DataContainer<data_t>{dd, bVec}};
        QuadricProblem<data_t> prob{scalingOp, dcB, true};

        WHEN("setting up a NLCG solver with Fletcher Reeves")
        {
            TestType solver{prob, TestType::Beta::FR};

            THEN("the clone works correctly")
            {
                auto nlcgClone = solver.clone();

                REQUIRE_NE(nlcgClone.get(), &solver);
                REQUIRE_EQ(*nlcgClone, solver);

                AND_THEN("it works as expected")
                {
                    auto solution = solver.solve(dd.getNumberOfCoefficients());

                    DataContainer<data_t> resultsDifference = scalingOp.apply(solution) - dcB;
                    // should have converged for the given number of iterations
                    REQUIRE_LE((resultsDifference).squaredL2Norm(),
                               epsilon * epsilon * dcB.squaredL2Norm());
                }
            }
        }

        WHEN("setting up a NLCG solver with Polak Riviere")
        {
            TestType solver{prob, TestType::Beta::PR};

            THEN("the clone works correctly")
            {
                auto nlcgClone = solver.clone();

                REQUIRE_NE(nlcgClone.get(), &solver);
                REQUIRE_EQ(*nlcgClone, solver);

                AND_THEN("it works as expected")
                {
                    auto solution = solver.solve(dd.getNumberOfCoefficients());

                    DataContainer<data_t> resultsDifference = scalingOp.apply(solution) - dcB;
                    // should have converged for the given number of iterations
                    REQUIRE_LE((resultsDifference).squaredL2Norm(),
                               epsilon * epsilon * dcB.squaredL2Norm());
                }
            }
        }
    }
}
TEST_SUITE_END();