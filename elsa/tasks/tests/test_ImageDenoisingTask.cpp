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

TEST_CASE_TEMPLATE("ImageDenoisingTask: Overfitting a random image", data_t, float, double)
{
    // eliminate the timing info from console for the tests
    // Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("a random image")
    {
        VolumeDescriptor imageDescriptor({5, 4});

        Vector_t<data_t> imageVec(imageDescriptor.getNumberOfCoefficients());
        imageVec.setRandom();
        DataContainer<data_t> image(imageDescriptor, imageVec);

        WHEN("setting up a ImageDenoisingTask with dictionary learning")
        {
            const index_t blocksize = 2;
            const index_t stride = 1;
            const index_t nAtoms = 7;
            const index_t sparsityLevel = 3;
            const index_t nIterations = 10;

            ImageDenoisingTask<data_t> denoiseTask(blocksize, stride, sparsityLevel, nAtoms,
                                                   nIterations);

            THEN("the original image can be reconstructed by overfitting")
            {
                auto reconstruction = denoiseTask.train(image);

                // with current setttings: intially 9 to <0.05
                // only require an error <0.1, can't expect too much when using a random signal
                REQUIRE_UNARY(isApprox(image, reconstruction, 0.1));
            }

            AND_THEN("the original image can be reconstructed by training with downsampling")
            {
                denoiseTask.train(image, 0.2);
                auto reconstruction = denoiseTask.denoise(image);

                // with current setttings: intially 9 to <0.05
                // only require an error <0.1, can't expect too much when using a random signal
                REQUIRE_UNARY(isApprox(image, reconstruction, 0.1));
            }
        }

        WHEN("setting up a ImageDenoisingTask with deep dictionary learning")
        {
            const index_t blocksize = 2;
            const index_t stride = 1;
            const std::vector<index_t> nAtoms{7, 6, 5};
            std::vector<ActivationFunction<data_t>> activations;
            activations.push_back(IdentityActivation<data_t>());
            activations.push_back(IdentityActivation<data_t>());
            const index_t sparsityLevel = 3;
            const index_t nIterations = 10;

            // ImageDenoisingTask<data_t> denoiseTask(blocksize, stride, sparsityLevel, nIterations,
            // nAtoms, activations);
            ImageDenoisingTask<data_t> denoiseTask(blocksize, stride, sparsityLevel, nIterations,
                                                   nAtoms);

            THEN("the original image can be reconstructed by overfitting")
            {
                auto reconstruction = denoiseTask.train(image);

                // with current setttings: intially 9 to <0.05
                // only require an error <0.1, can't expect too much when using a random signal
                REQUIRE_UNARY(isApprox(image, reconstruction, 0.1));
            }

            /*
                        AND_THEN("the original image can be reconstructed by training with
               downsampling")
                        {
                            denoiseTask.train(image, 0.2);
                            auto reconstruction = denoiseTask.denoise(image);

                            // with current setttings: intially 9 to <0.05
                            // only require an error <0.1, can't expect too much when using a random
               signal REQUIRE_UNARY(isApprox(image, reconstruction, 0.1));
                        }
            */
        }
    }
}

TEST_SUITE_END();
