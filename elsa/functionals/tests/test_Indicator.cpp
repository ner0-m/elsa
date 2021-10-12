/**
 * @file test_Indicator.cpp
 *
 * @brief Tests for the Indicator class
 *
 * @author Andi Braimllari
 */

#include "Indicator.h"
#include "VolumeDescriptor.h"
#include "testHelpers.h"
#include "TypeCasts.hpp"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("functionals");

TEST_CASE_TEMPLATE("Indicator: Testing the indicator functional", TestType, float, double)
{
    using Vector = Eigen::Matrix<TestType, Eigen::Dynamic, 1>;

    GIVEN("just data (no residual)")
    {
        IndexVector_t numCoeff(1);
        numCoeff << 4;
        VolumeDescriptor volDescr(numCoeff);

        WHEN("instantiating")
        {
            Indicator<TestType> indicator(volDescr, std::less_equal<TestType>(), 0);

            THEN("the indicator is as expected")
            {
                REQUIRE_EQ(indicator.getDomainDescriptor(), volDescr);
            }

            THEN("a clone behaves as expected")
            {
                auto indicatorClone = indicator.clone();

                REQUIRE(indicatorClone.get() != &indicator);
                REQUIRE(*indicatorClone == indicator);
            }

            THEN("the evaluate returns 0")
            {
                Vector dataVec(volDescr.getNumberOfCoefficients());
                dataVec << 5, 2, 1, 1;
                DataContainer<TestType> dc(volDescr, dataVec);

                REQUIRE(checkApproxEq(indicator.evaluate(dc), 0));
            }

            THEN("the evaluate returns +infinity")
            {
                Vector dataVec(volDescr.getNumberOfCoefficients());
                dataVec << -1, 7, 3, 2;
                DataContainer<TestType> dc(volDescr, dataVec);

                REQUIRE_UNARY(indicator.evaluate(dc) == std::numeric_limits<TestType>::infinity());
            }

            THEN("gradient and Hessian work as expected")
            {
                Vector dataVec(volDescr.getNumberOfCoefficients());
                dataVec << -3, -4, 1, 9;
                DataContainer<TestType> dc(volDescr, dataVec);

                REQUIRE_THROWS_AS(indicator.getGradient(dc), LogicError);
                REQUIRE_THROWS_AS(indicator.getHessian(dc), LogicError);
            }
        }
    }
}

TEST_SUITE_END();
