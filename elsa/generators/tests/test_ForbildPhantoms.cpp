/**
 * @file test_Phantoms.cpp
 *
 * @brief Tests for the Phantoms class
 *
 * @author Tobias Lasser - nothing to see here...
 */

#include "doctest/doctest.h"
#include "Phantoms.h"
#include "VolumeDescriptor.h"

using namespace elsa;
using namespace doctest;

TEST_CASE("Forbild Phantoms")
{
    GIVEN("Head phantom")
    {
        IndexVector_t sizes(3);
        sizes << 64, 64, 64;

        WHEN("rasterize head")
        {
            auto dc = phantoms::forbildHead<real_t>(sizes);

            THEN("it looks good")
            {
                REQUIRE(true); // TODO: implement checks to validate voxels
            }
        }
    }

    GIVEN("Abdomen phantom")
    {
        IndexVector_t sizes(3);
        sizes << 64, 64, 64;

        WHEN("rasterize abdomen")
        {
            auto dc = phantoms::forbildAbdomen<real_t>(sizes);

            THEN("it looks good")
            {
                REQUIRE(true); // TODO: implement checks to validate voxels
            }
        }
    }

    GIVEN("Thorax phantom")
    {
        IndexVector_t sizes(3);
        sizes << 64, 64, 64;

        WHEN("rasterize thorax")
        {
            auto dc = phantoms::forbildThorax<real_t>(sizes);

            THEN("it looks good")
            {
                REQUIRE(true); // TODO: implement checks to validate voxels
            }
        }
    }
}
