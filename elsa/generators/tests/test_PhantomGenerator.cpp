/**
 * @file test_PhantomGenerator.cpp
 *
 * @brief Tests for the PhantomGenerator class
 *
 * @author Tobias Lasser - nothing to see here...
 */

#include "doctest/doctest.h"
#include "PhantomGenerator.h"
#include "testHelpers.h"

using namespace elsa;
using namespace doctest;

RealVector_t get2dModifiedSheppLogan45x45();

TEST_CASE("PhantomGenerator: Drawing a 2d Shepp-Logan phantom")
{
    GIVEN("a volume size")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 512, 512;

        WHEN("creating a 2d Shepp-Logan")
        {
            auto dc = PhantomGenerator<real_t>::createModifiedSheppLogan(numCoeff);

            THEN("it looks good")
            {
                REQUIRE(true); // TODO: add a proper test here
            }
        }
    }
}

TEST_CASE("PhantomGenerator: Drawing a 3d Shepp-Logan phantom")
{
    GIVEN("a volume size")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 64, 64, 64;

        WHEN("creating a 3d Shepp-Logan")
        {
            auto dc = PhantomGenerator<real_t>::createModifiedSheppLogan(numCoeff);

            THEN("it looks good")
            {
                REQUIRE(true); // TODO: add a proper test here
            }
        }
    }
}
