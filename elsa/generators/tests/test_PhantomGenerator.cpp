/**
 * \file test_PhantomGenerator.cpp
 *
 * \brief Tests for the PhantomGenerator class
 *
 * \author Tobias Lasser - nothing to see here...
 */

#include <catch2/catch.hpp>
#include "PhantomGenerator.h"

using namespace elsa;

RealVector_t get2dModifiedSheppLogan45x45();

SCENARIO("Drawing a 2d Shepp-Logan phantom") {
    GIVEN("a volume size") {
        IndexVector_t numCoeff(2); numCoeff << 512, 512;

        WHEN("creating a 2d Shepp-Logan") {
            auto dc = PhantomGenerator<real_t>::createModifiedSheppLogan(numCoeff);

            THEN("it looks good") {
                REQUIRE(true); // TODO: add a proper test here
            }
        }
    }
}


SCENARIO("Drawing a 3d Shepp-Logan phantom") {
    GIVEN("a volume size") {
        IndexVector_t numCoeff(3); numCoeff << 64, 64, 64;

        WHEN("creating a 3d Shepp-Logan") {
            auto dc = PhantomGenerator<real_t>::createModifiedSheppLogan(numCoeff);

            THEN("it looks good") {
                REQUIRE(true); // TODO: add a proper test here
            }
        }
    }
}
