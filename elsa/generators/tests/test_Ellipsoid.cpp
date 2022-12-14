/**
 * @file test_Sphere.cpp
 *
 * @brief Test for Sphere
 *
 */

#include "doctest/doctest.h"

#include "Ellipsoid.h"

using namespace elsa;
using namespace doctest;

TEST_CASE("Ellipsoid tests")
{
    GIVEN("A Ellipsoid - no rotation")
    {
        index_t size = 10;
        IndexVector_t sizes(3);
        sizes << size, size, size;

        double amplit = 1.0;

        phantoms::Vec3i center(3);
        center << 5.0, 5.0, 5.0;

        phantoms::Vec3X<double> halfAxis;
        halfAxis << 1., 2., 3.;

        phantoms::Vec3X<double> eulers;
        eulers << 0., 0., 0.;

        VolumeDescriptor dd(sizes);
        DataContainer<double> dc(dd);
        dc = 0;

        WHEN("Rasterize ellipsoid")
        {
            phantoms::Ellipsoid<double> s{amplit, center, halfAxis, eulers};
            phantoms::rasterize<double>(s, dd, dc);

            IndexVector_t idx(3);

            WHEN("Check empty volume")
            {
                idx << 3, 5, 5;
                REQUIRE_EQ(dc(idx), 0);

                idx << 5, 8, 5;
                REQUIRE_EQ(dc(idx), 0);

                idx << 5, 5, 9;
                REQUIRE_EQ(dc(idx), 0);
            }
            WHEN("Check filled volume")
            {
                idx << 5, 5, 5;
                REQUIRE_EQ(dc(idx), amplit);
                idx << 6, 5, 5;
                REQUIRE_EQ(dc(idx), amplit);
                idx << 5, 3, 5;
                REQUIRE_EQ(dc(idx), amplit);
                idx << 5, 5, 2;
                REQUIRE_EQ(dc(idx), amplit);
            }
        }
    }

    GIVEN("A Ellipsoid - with rotation")
    {
        index_t size = 10;
        IndexVector_t sizes(3);
        sizes << size, size, size;

        double amplit = 1.0;

        phantoms::Vec3i center(3);
        center << 5.0, 5.0, 5.0;

        phantoms::Vec3X<double> halfAxis;
        halfAxis << 1., 2., 3.;

        phantoms::Vec3X<double> eulers;
        eulers << 90., 0., 0.;

        VolumeDescriptor dd(sizes);
        DataContainer<double> dc(dd);
        dc = 0;

        WHEN("Rasterize ellipsoid")
        {
            phantoms::Ellipsoid<double> s{amplit, center, halfAxis, eulers};
            phantoms::rasterize<double>(s, dd, dc);

            IndexVector_t idx(3);

            WHEN("Check empty volume")
            {
                idx << 2, 5, 5;
                REQUIRE_EQ(dc(idx), 0);

                idx << 5, 3, 5;
                REQUIRE_EQ(dc(idx), 0);

                idx << 5, 5, 9;
                REQUIRE_EQ(dc(idx), 0);
            }
            WHEN("Check filled volume")
            {
                idx << 5, 5, 5;
                REQUIRE_EQ(dc(idx), amplit);
                idx << 3, 5, 5;
                REQUIRE_EQ(dc(idx), amplit);
                idx << 5, 4, 5;
                REQUIRE_EQ(dc(idx), amplit);
                idx << 5, 5, 2;
                REQUIRE_EQ(dc(idx), amplit);
            }
        }
    }

    GIVEN("A Ellipsoid - check Clipping")
    {
        index_t size = 10;
        IndexVector_t sizes(3);
        sizes << size, size, size;

        double amplit = 1.0;

        phantoms::Vec3i center(3);
        center << 5.0, 5.0, 5.0;

        phantoms::Vec3X<double> halfAxis;
        halfAxis << 5., 5., 5.;

        phantoms::Vec3X<double> eulers;
        eulers << 0., 0., 0.;

        VolumeDescriptor dd(sizes);
        DataContainer<double> dc(dd);
        dc = 0;

        auto clipZ = [](std::array<double, 6> minMax) {
            // limit maxZ to 0 in object space -> only render half ellipsoid
            minMax[5] = 0;
            return minMax;
        };

        WHEN("Rasterize ellipsoid")
        {
            phantoms::Ellipsoid<double> s{amplit, center, halfAxis, eulers};

            phantoms::rasterizeWithClipping<double>(s, dd, dc, clipZ);

            for (index_t i = 0; i <= center[phantoms::INDEX_Z]; i++) {
                // Rasterization "in front of" the ellipsoid center along the Z-Axis
                INFO("Check Slice  ", i);
                REQUIRE_EQ(dc.slice(i).maxElement(), amplit);
            }

            for (index_t i = center[phantoms::INDEX_Z] + 1; i < sizes[phantoms::INDEX_Z]; i++) {
                // No rasterization "behind" the ellipsoid center along the Z-Axis
                INFO("Check Slice  ", i);
                REQUIRE_EQ(dc.slice(i).maxElement(), 0);
            }
        }
    }
}
