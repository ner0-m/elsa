/**
 * @file test_ImageDenoisingTask.cpp
 *
 * @brief Tests for the ImageDenoisingTask class
 *
 * @author Jonas Buerger
 */

#include "doctest/doctest.h"

#include <limits>
#include "ImageDenoisingTask.h"
#include "VolumeDescriptor.h"
#include "testHelpers.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("tasks");

TEST_CASE("ImageDenoisingTask: Overfitting a random image")
{
    // eliminate the timing info from console for the tests
    // Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("a random image")
    {
        VolumeDescriptor imageDescriptor({5, 4});

        RealVector_t imageVec(imageDescriptor.getNumberOfCoefficients());
        imageVec.setRandom();
        DataContainer image(imageDescriptor, imageVec);

        WHEN("setting up a ImageDenoisingTask")
        {
            const index_t blocksize = 2;
            const index_t stride = 1;
            const index_t nAtoms = 7;
            const index_t sparsityLevel = 3;
            const index_t nIterations = 100;

            ImageDenoisingTask denoiseTask(blocksize, stride, sparsityLevel, nAtoms, nIterations);

            THEN("the original image can be reconstructed by overfitting")
            {
                auto reconstruction = denoiseTask.train(image);

                // with current setttings: intially 9 to <0.05
                REQUIRE_EQ(image, reconstruction);
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
