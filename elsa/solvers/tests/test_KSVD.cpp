/**
 * @file test_KSVD.cpp
 *
 * @brief Tests for the KSVD class
 *
 * @author Jonas Buerger
 */

#include "doctest/doctest.h"

#include <limits>
#include "KSVD.h"
#include "VolumeDescriptor.h"
#include "testHelpers.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("solvers");

TEST_CASE("KSVD: Solving a DictionaryLearningProblem")
{
    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("a random signal")
    {
        VolumeDescriptor dd({5});
        IdenticalBlocksDescriptor signalsDescriptor(1, dd);
        const index_t nAtoms = 15;

        RealVector_t signalVec(dd.getNumberOfCoefficients());
        signalVec.setRandom();
        DataContainer signals(signalsDescriptor, signalVec);

        DictionaryLearningProblem dictProb(signals, nAtoms);

        WHEN("setting up a DictionaryLearningProblem from it")
        {
            DictionaryLearningProblem dictProb(signals, nAtoms);
            KSVD solver(dictProb);

            THEN("a suitable dictionary and representation are found")
            {
                auto solution = solver.solve(20);
                auto& learnedDict = solver.getLearnedDictionary();

                REQUIRE_UNARY(
                    isApprox(signals.getBlock(0), learnedDict.apply(solution.getBlock(0))));
            }
        }
    }
}

TEST_SUITE_END();
