/**
 * \file test_elsaDefines.cpp
 *
 * \brief Tests for common elsa defines
 *
 * \author David Frank - initial version
 */

#include <catch2/catch.hpp>
#include "elsaDefines.h"

using namespace elsa;

SCENARIO("Testing PI")
{

    THEN("Pi for real_t and pi_t are equal") { REQUIRE(pi<real_t> == pi_t); }

    THEN("pi_t is somewhat close to a representation for pi")
    {
        REQUIRE(pi_t == Approx(3.14159265358979323846).epsilon(1e-5));
    }

    THEN("Pi for double is close to given value for pi")
    {
        REQUIRE(pi<double> == 3.14159265358979323846);
    }
}