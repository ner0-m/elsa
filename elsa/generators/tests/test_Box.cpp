/**
 * @file test_box.cpp
 *
 * @brief Test for Box
 *
 */

#include "doctest/doctest.h"

#include "PhantomDefines.h"
#include "Box.h"
#include <iostream>

using namespace elsa;
using namespace doctest;

TEST_CASE("Box tests")
{
    GIVEN("A shifted Box")
    {
        index_t size = 10;
        IndexVector_t sizes(3);
        sizes << size, size, size;

        phantoms::Vec3i center;
        center << 5.0, 5.0, 5.0;

        phantoms::Vec3X<double> edgeLength;
        edgeLength << 2, 4, 6;

        VolumeDescriptor dd(sizes);
        DataContainer<double> dc(dd);
        dc = 0;

        WHEN("Rasterize box")
        {
            phantoms::Box<double> s{1.0, center, edgeLength};
            phantoms::rasterize<phantoms::Blending::ADDITION>(s, dd, dc);

            IndexVector_t idx(3);

            WHEN("Check empty volume")
            {
                idx << 1, 5, 5;
                REQUIRE_EQ(dc(idx), 0);

                idx << 4, 2, 1;
                REQUIRE_EQ(dc(idx), 0);

                idx << 6, 7, 9;
                REQUIRE_EQ(dc(idx), 0);
            }
            WHEN("Check filled volume")
            {
                idx << 5, 5, 5;
                REQUIRE_EQ(dc(idx), 1.0);
                idx << 5, 6, 6;
                REQUIRE_EQ(dc(idx), 1.0);
                idx << 6, 7, 8;
                REQUIRE_EQ(dc(idx), 1.0);
            }
        }
    }
}
