/**
 * @file test_Patchifier.cpp
 *
 * @brief Tests for Patchifier class
 *
 * @author Jonas Buerger - initial code
 */

#include "doctest/doctest.h"
#include "Error.h"
#include "IdenticalBlocksDescriptor.h"
#include "VolumeDescriptor.h"
#include "Patchifier.h"
#include <stdexcept>

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("core");

TEST_CASE_TEMPLATE("Patchifier: Using the patchifier", data_t, float, double)
{
    GIVEN("An image and a corresponding patchifier with stride 1")
    {

        VolumeDescriptor imageDescriptor({5, 4});
        index_t blockSize = 3;
        index_t stride = 1;
        Patchifier<data_t> patchifier(imageDescriptor, blockSize, stride);

        Vector_t<data_t> imageVec(imageDescriptor.getNumberOfCoefficients());
        imageVec << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20;
        DataContainer<data_t> dcImage(imageDescriptor, imageVec);

        WHEN("turning the image into patches")
        {
            auto patches = patchifier.im2patches(dcImage);
            THEN("the patches are as expected")
            {
                IdenticalBlocksDescriptor patchesDescriptor(6,
                                                            VolumeDescriptor{blockSize, blockSize});

                REQUIRE_EQ(patchesDescriptor,
                           downcast_safe<IdenticalBlocksDescriptor>(patches.getDataDescriptor()));

                Vector_t<data_t> patchesVec(patchesDescriptor.getNumberOfCoefficients());
                patchesVec << 1, 2, 3, 6, 7, 8, 11, 12, 13, // patch 1
                    2, 3, 4, 7, 8, 9, 12, 13, 14,           // patch 2
                    3, 4, 5, 8, 9, 10, 13, 14, 15,          // patch 3
                    6, 7, 8, 11, 12, 13, 16, 17, 18,        // patch 4
                    7, 8, 9, 12, 13, 14, 17, 18, 19,        // patch 5
                    8, 9, 10, 13, 14, 15, 18, 19, 20;       // patch 6
                DataContainer<data_t> expected(patchesDescriptor, patchesVec);

                REQUIRE_EQ(patches, expected);
            }
        }

        WHEN("turning the image into patches and the result back to an image")
        {
            auto patches = patchifier.im2patches(dcImage);
            auto img = patchifier.patches2im(patches);

            THEN("the result equals the original image") { REQUIRE_EQ(img, dcImage); }
        }
    }

    GIVEN("An image and a corresponding patchifier with stride 2")
    {
        // it is important that blocksize, stride and image size fit together, otherwise the pixels
        // at the border will be truncated and the depatchified image doesnt match the original
        VolumeDescriptor imageDescriptor({7, 5});
        index_t blockSize = 3;
        index_t stride = 2;
        Patchifier<data_t> patchifier(imageDescriptor, blockSize, stride);

        Vector_t<data_t> imageVec(imageDescriptor.getNumberOfCoefficients());
        imageVec.setRandom();
        DataContainer<data_t> dcImage(imageDescriptor, imageVec);

        WHEN("turning the image into patches and the result back to an image")
        {
            auto patches = patchifier.im2patches(dcImage);
            auto img = patchifier.patches2im(patches);

            THEN("the result equals the original image") { REQUIRE_EQ(img, dcImage); }
        }
    }
}
TEST_SUITE_END();
