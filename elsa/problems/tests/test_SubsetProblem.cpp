/**
 * @file test_SubsetProblem.cpp
 *
 * @brief Tests for the SubsetProblem class
 *
 * @author Michael Loipführer - initial code
 */

#include "doctest/doctest.h"

#include "Logger.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("problems");

TEST_CASE_TEMPLATE("SubsetProblem: Empty test", data_t, float, double)
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

TEST_SUITE_END();
