/**
 * \file test_Timer.cpp
 *
 * \brief Tests for Timer class
 *
 * \author Maximilian Hornung
 */

#include <catch2/catch.hpp>
#include "Timer.h"

using namespace elsa;

SCENARIO("Using Timer") {
    Timer("who", "what");
    // cannot test much more than creation/destruction...

    REQUIRE(true);
}