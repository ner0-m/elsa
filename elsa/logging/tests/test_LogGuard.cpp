/**
 * \file test_LogGuard.cpp
 *
 * \brief Tests for LogGuard class
 *
 * \author Maximilian Hornung
 */

#include <catch2/catch.hpp>
#include "LogGuard.h"

using namespace elsa;

SCENARIO("Using LogGuard")
{
    LogGuard("logger", "message");
    // cannot test much more than creation/destruction...

    REQUIRE(true);
}