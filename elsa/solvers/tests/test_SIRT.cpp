
/**
 * @file test_SIRT.cpp
 *
 * @brief Tests for the SIRT class
 *
 * @author Maryna Shcherbak
 */

#include "doctest/doctest.h"
#include "SIRT.h"
#include "Identity.h"
#include "LinearResidual.h"
#include "Logger.h"
#include "VolumeDescriptor.h"
#include "testHelpers.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("solvers");

TYPE_TO_STRING(SIRT<float>);
TYPE_TO_STRING(SIRT<double>);

template <template <typename> typename T, typename data_t>
constexpr data_t return_data_t(const T<data_t>&);

TEST_CASE_TEMPLATE("Scenario: Solving a simple linear problem", TestType, SIRT<float>, SIRT<double>)
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

        WHEN("setting up a SIRT solver with fixed step size")
        {
            TestType solver(prob);

            THEN("the clone works correctly")
            {
                auto sirtClone = solver.clone();

                REQUIRE_NE(sirtClone.get(), &solver);
                REQUIRE_EQ(*sirtClone, solver);

                AND_THEN("it works as expected")
                {
                    auto solution = solver.solve(1000);
                    REQUIRE_UNARY(checkApproxEq(solution.squaredL2Norm(), bVec.squaredNorm()));
                }
            }
        }
    }
}

TEST_SUITE_END();