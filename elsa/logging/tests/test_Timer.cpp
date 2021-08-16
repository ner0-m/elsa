/**
 * @file test_Timer.cpp
 *
 * @brief Tests for Timer class
 *
 * @author Maximilian Hornung
 */

#include "doctest/doctest.h"
#include "Timer.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("logging");

TEST_CASE("Timer: Testing usage")
{
    Timer("who", "what");
    // cannot test much more than creation/destruction...

    REQUIRE(true);
}

TEST_SUITE_END();
