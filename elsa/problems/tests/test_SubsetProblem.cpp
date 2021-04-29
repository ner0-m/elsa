/**
 * @file test_SubsetProblem.cpp
 *
 * @brief Tests for the SubsetProblem class
 *
 * @author Michael Loipführer - initial code
 */

#include <catch2/catch.hpp>
#include "Logger.h"

using namespace elsa;

TEMPLATE_TEST_CASE("Scenario: Testing SubsetProblem", "", float, double)
{
    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::WARN);

    GIVEN("A small ordered subset problem")
    {
        // TODO: implement actual minimal tests
        // This is quite hard as the default operators (Identity, Scaling) do not allow for
        // their domain descriptor to differ from their range descriptor and therefore cant be
        // split up into subsets. Tests using the Phantom generator, an actual projector and a
        // SubsetSampler would work but require a whole bunch of other components.
    }
}