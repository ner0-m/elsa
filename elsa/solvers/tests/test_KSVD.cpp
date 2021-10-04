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
        const index_t nAtoms = 10;

        RealVector_t signalVec(dd.getNumberOfCoefficients());
        signalVec.setRandom();
        DataContainer signals(signalsDescriptor, signalVec);

        WHEN("setting up a DictionaryLearningProblem from it")
        {
            DictionaryLearningProblem dictProb(signals, nAtoms);
            KSVD solver(dictProb, 3);

            THEN("a suitable dictionary and representation are found")
            {
                auto solution = solver.solve(5);
                auto& learnedDict = solver.getLearnedDictionary();

                REQUIRE_UNARY(
                    isApprox(signals.getBlock(0), learnedDict.apply(solution.getBlock(0))));
            }
        }
    }

    GIVEN("Multiple random signals")
    {
        VolumeDescriptor dd({5});
        const index_t nAtoms = 10;

        RealVector_t signalVec1(dd.getNumberOfCoefficients());
        RealVector_t signalVec2(dd.getNumberOfCoefficients());
        RealVector_t signalVec3(dd.getNumberOfCoefficients());
        signalVec1.setRandom();
        signalVec2.setRandom();
        signalVec3.setRandom();

        IdenticalBlocksDescriptor signalsDescriptor(3, dd);
        DataContainer signals(signalsDescriptor);
        signals.getBlock(0) = DataContainer(dd, signalVec1);
        signals.getBlock(1) = DataContainer(dd, signalVec2);
        signals.getBlock(2) = DataContainer(dd, signalVec3);

        WHEN("setting up a DictionaryLearningProblem from them")
        {
            DictionaryLearningProblem dictProb(signals, nAtoms);
            KSVD solver(dictProb, 3);

            THEN("a suitable dictionary and representation are found")
            {
                auto solution = solver.solve(10);
                auto& learnedDict = solver.getLearnedDictionary();

                for (index_t i = 0; i < 3; ++i) {
                    REQUIRE_UNARY(
                        isApprox(signals.getBlock(i), learnedDict.apply(solution.getBlock(i))));
                }
            }
        }
    }
}

TEST_SUITE_END();
