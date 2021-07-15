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

TEST_CASE_TEMPLATE("Testing the indicator functional", TestType, float, double)
{
    using Vector = Eigen::Matrix<TestType, Eigen::Dynamic, 1>;

    GIVEN("just data (no residual)")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 7, 17;
        VolumeDescriptor volDescr(numCoeff);

        WHEN("instantiating")
        {
            Indicator<TestType> indicator(volDescr);

            THEN("a clone behaves as expected")
            {
                auto indicatorClone = indicator.clone();

                REQUIRE(indicatorClone.get() != &indicator);
                REQUIRE(*indicatorClone == indicator);
            }
        }
    }
}

TEST_SUITE_END();
