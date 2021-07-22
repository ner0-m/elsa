/**
 * \file test_Cimmino.cpp
 *
 * \brief Tests for the Cimmino class
 *
 * \author Maryna Shcherbak - initial code
 */

#include "doctest/doctest.h"
#include "Cimmino.h"
#include "WLSProblem.h"
#include "Problem.h"
#include "Identity.h"
#include "LinearResidual.h"
#include "Logger.h"
#include "VolumeDescriptor.h"
#include "testHelpers.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("solvers");

TYPE_TO_STRING(Cimmino<float>);
TYPE_TO_STRING(Cimmino<double>);

template <template <typename> typename T, typename data_t>
constexpr data_t return_data_t(const T<data_t>&);

TEST_CASE_TEMPLATE("Scenario: Solving a simple linear problem", TestType, Cimmino<float>,
                   Cimmino<double>)
{
    // Set seed for Eigen Matrices!
    srand((unsigned int) 666);

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

        WHEN("setting up a Cimmino solver with fixed relaxation parameter")
        {
            TestType solver{prob, 1.1};

            THEN("the clone works correctly")
            {
                auto cClone = solver.clone();

                REQUIRE_NE(cClone.get(), &solver);
                REQUIRE_EQ(*cClone, solver);

                AND_THEN("it works as expected")
                {
                    auto solution = solver.solve(2000);
                    REQUIRE_UNARY(checkApproxEq(solution.squaredL2Norm(), bVec.squaredNorm()));
                }
            }
        }

        WHEN("setting up a Cimmino solver with a default step size")
        {
            TestType solver{prob};

            THEN("the clone works correctly")
            {
                auto cClone = solver.clone();

                REQUIRE_NE(cClone.get(), &solver);
                REQUIRE_EQ(*cClone, solver);

                AND_THEN("it works as expected")
                {
                    auto solution = solver.solve(2000);
                    REQUIRE_UNARY(checkApproxEq(solution.squaredL2Norm(), bVec.squaredNorm()));
                }
            }
        }
    }
}

TEST_SUITE_END();