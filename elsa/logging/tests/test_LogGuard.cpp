/**
 * @file test_LogGuard.cpp
 *
 * @brief Tests for LogGuard class
 *
 * @author Maximilian Hornung
 */

#include "doctest/doctest.h"
#include "LogGuard.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("logging");

TEST_CASE("LogGuard: Testing usage")
{
    LogGuard("logger", "message");
    // cannot test much more than creation/destruction...

    REQUIRE_UNARY(true);
}

TEST_SUITE_END();
