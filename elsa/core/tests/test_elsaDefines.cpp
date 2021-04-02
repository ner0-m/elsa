/**
 * @file test_elsaDefines.cpp
 *
 * @brief Tests for common elsa defines
 *
 * @author David Frank - initial version
 */

#include "doctest/doctest.h"
#include <iostream>
#include "elsaDefines.h"
#include "Logger.h"
#include <Eigen/Core>

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("core");

TEST_CASE("elsaDefines: Testing PI")
{

    THEN("Pi for real_t and pi_t are equal") { REQUIRE_EQ(pi<real_t>, pi_t); }

    THEN("pi_t is somewhat close to a representation for pi")
    {
        REQUIRE_EQ(pi_t, Approx(3.14159265358979323846).epsilon(1e-5));
    }

    THEN("Pi for double is close to given value for pi")
    {
        REQUIRE_EQ(pi<double>, 3.14159265358979323846);
    }
}

TEST_CASE("elsaDefines: Testing compile-time predicates")
{
    static_assert(std::is_same_v<float, GetFloatingPointType_t<std::complex<float>>>);
    static_assert(std::is_same_v<double, GetFloatingPointType_t<std::complex<double>>>);
    static_assert(std::is_same_v<double, GetFloatingPointType_t<double>>);
    static_assert(std::is_same_v<float, GetFloatingPointType_t<float>>);
    static_assert(!std::is_same_v<float, GetFloatingPointType_t<double>>);

    REQUIRE_UNARY(true);
}

TEST_CASE("elsaDefines: Printing default handler type")
{
#ifdef ELSA_CUDA_VECTOR
    REQUIRE(defaultHandlerType == DataHandlerType::GPU);
#else
    REQUIRE(defaultHandlerType == DataHandlerType::CPU);
#endif
}

TEST_CASE("elsaDefines: Printing Eigen Matrix")
{
    // FIXME: Actually test the formatting, this only tests compilation
    Eigen::Matrix<real_t, Eigen::Dynamic, 1> vec(3);
    vec << 1, 2, 4.5;

    infoln("Printing without testing: {}", vec);
}


TEST_SUITE_END();
