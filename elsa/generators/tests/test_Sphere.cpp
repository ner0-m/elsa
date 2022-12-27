/**
 * @file test_Sphere.cpp
 *
 * @brief Test for Sphere
 *
 */

#include "doctest/doctest.h"

#include "PhantomDefines.h"
#include "Sphere.h"

using namespace elsa;
using namespace doctest;

TEST_CASE("Sphere tests")
{
    GIVEN("A shifted sphere")
    {
        index_t size = 10;
        IndexVector_t sizes(3);
        sizes << size, size, size;

        phantoms::Vec3i center(3);
        center << 5.0, 5.0, 5.0;

        double radius = 2.5;

        VolumeDescriptor dd(sizes);
        DataContainer<double> dc(dd);
        dc = 0;

        WHEN("Rasterize sphere")
        {
            phantoms::Sphere<double> s{1.0, center, radius};
            phantoms::rasterize<phantoms::Blending::ADDITION>(s, dd, dc);

            IndexVector_t idx(3);

            WHEN("Check empty volume")
            {
                idx << 2, 5, 5;
                REQUIRE_EQ(dc(idx), 0);

                idx << 5, 2, 5;
                REQUIRE_EQ(dc(idx), 0);

                idx << 5, 5, 2;
                REQUIRE_EQ(dc(idx), 0);
            }
            WHEN("Check filled volume")
            {
                idx << 5, 5, 5;
                REQUIRE_EQ(dc(idx), 1.0);
                idx << 3, 5, 5;
                REQUIRE_EQ(dc(idx), 1.0);
                idx << 6, 6, 6;
                REQUIRE_EQ(dc(idx), 1.0);
            }
        }
    }
}
