/**
 * @file test_EllipCylinder.cpp
 *
 * @brief Test for EllipCylinder
 *
 */

#include "doctest/doctest.h"

#include "PhantomDefines.h"
#include "Cylinder.h"

using namespace elsa;
using namespace doctest;

TEST_CASE("cylinder tests")
{
    GIVEN("A shifted cylinder on z axis")
    {
        index_t size = 10;
        IndexVector_t sizes(3);
        sizes << size, size, size;

        phantoms::Vec3i center;
        center << 5.0, 5.0, 5.0;

        double length{3.};
        double amplit{1.};
        double radius{1.};

        VolumeDescriptor dd(sizes);
        DataContainer<double> dc(dd);
        dc = 0;

        WHEN("Rasterize box")
        {
            phantoms::Cylinder<double> cyl{elsa::phantoms::Orientation::Z_AXIS, amplit, center,
                                           radius, length};

            phantoms::rasterize<phantoms::Blending::ADDITION>(cyl, dd, dc);

            IndexVector_t idx(3);

            WHEN("Check empty volume")
            {
                idx << 1, 5, 5;
                REQUIRE_EQ(dc(idx), 0);

                idx << 9, 5, 5;
                REQUIRE_EQ(dc(idx), 0);
            }
            WHEN("Check filled volume")
            {

                idx << 5, 5, 5;
                REQUIRE_EQ(dc(idx), amplit);

                idx << 5, 5, 6;
                REQUIRE_EQ(dc(idx), amplit);

                idx << 5, 5, 3;
                REQUIRE_EQ(dc(idx), amplit);
            }
        }
    }

    GIVEN("A shifted EllipCylinder on y axis")
    {

        index_t size = 10;
        IndexVector_t sizes(3);
        sizes << size, size, size;

        phantoms::Vec3i center;
        center << 5.0, 5.0, 5.0;

        double length{3.};
        double amplit{1.};
        double radius{1.};

        VolumeDescriptor dd(sizes);
        DataContainer<double> dc(dd);
        dc = 0;

        WHEN("Rasterize box")
        {
            phantoms::Cylinder<double> cyl{elsa::phantoms::Orientation::Y_AXIS, amplit, center,
                                           radius, length};

            phantoms::rasterize<phantoms::Blending::ADDITION>(cyl, dd, dc);

            IndexVector_t idx(3);

            WHEN("Check empty volume")
            {
                idx << 1, 5, 5;
                REQUIRE_EQ(dc(idx), 0);

                idx << 9, 5, 5;
                REQUIRE_EQ(dc(idx), 0);
            }
            WHEN("Check filled volume")
            {

                idx << 5, 5, 5;
                REQUIRE_EQ(dc(idx), amplit);

                idx << 5, 6, 5;
                REQUIRE_EQ(dc(idx), amplit);

                idx << 5, 3, 5;
                REQUIRE_EQ(dc(idx), amplit);
            }
        }
    }
    GIVEN("A shifted EllipCylinder on x axis")
    {

        index_t size = 10;
        IndexVector_t sizes(3);
        sizes << size, size, size;

        phantoms::Vec3i center;
        center << 5.0, 5.0, 5.0;

        double length{3.};
        double amplit{1.};
        double radius{1.};

        VolumeDescriptor dd(sizes);
        DataContainer<double> dc(dd);
        dc = 0;

        WHEN("Rasterize box")
        {
            phantoms::Cylinder<double> cyl{elsa::phantoms::Orientation::X_AXIS, amplit, center,
                                           radius, length};

            phantoms::rasterize<phantoms::Blending::ADDITION>(cyl, dd, dc);

            IndexVector_t idx(3);

            WHEN("Check empty volume")
            {
                idx << 5, 1, 5;
                REQUIRE_EQ(dc(idx), 0);

                idx << 5, 9, 5;
                REQUIRE_EQ(dc(idx), 0);
            }
            WHEN("Check filled volume")
            {

                idx << 5, 5, 5;
                REQUIRE_EQ(dc(idx), amplit);

                idx << 6, 5, 5;
                REQUIRE_EQ(dc(idx), amplit);

                idx << 3, 5, 5;
                REQUIRE_EQ(dc(idx), amplit);
            }
        }
    }
}