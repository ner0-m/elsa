/**
 * @file CylinderFree.cpp
 *
 * @brief Test for CylinderFree
 *
 */

#include "doctest/doctest.h"

#include "PhantomDefines.h"
#include "CylinderFree.h"

using namespace elsa;
using namespace doctest;

TEST_CASE("CylinderFree tests")
{
    GIVEN("A rotated CylinderFree ")
    {
        index_t size = 10;
        IndexVector_t sizes(3);
        sizes << size, size, size;

        phantoms::Vec3i center;
        center << 5.0, 5.0, 5.0;

        double radius = 1.0;

        double length{8.};
        double amplit{1.};

        phantoms::Vec3X<double> eulers;
        eulers << 0.0, 45.0, 0.0;

        VolumeDescriptor dd(sizes);
        DataContainer<double> dc(dd);
        dc = 0;

        WHEN("Rasterize")
        {

            phantoms::CylinderFree<double> ec{amplit, center, radius, length, eulers};

            phantoms::rasterize<phantoms::Blending::ADDITION>(ec, dd, dc);

            IndexVector_t idx(3);

            WHEN("Check empty volume")
            {
                idx << 5, 5, 9;
                REQUIRE_EQ(dc(idx), 0);

                idx << 5, 5, 2;
                REQUIRE_EQ(dc(idx), 0);
            }
            WHEN("Check filled volume")
            {

                idx << 5, 5, 5;
                REQUIRE_EQ(dc(idx), amplit);

                idx << 5, 6, 6;
                REQUIRE_EQ(dc(idx), amplit);

                idx << 5, 4, 4;
                REQUIRE_EQ(dc(idx), amplit);
            }
        }
    }
}
