/**
 * @file test_EllipCylinder.cpp
 *
 * @brief Test for EllipCylinder
 *
 */

#include "doctest/doctest.h"

#include "PhantomDefines.h"
#include "EllipCylinder.h"

using namespace elsa;
using namespace doctest;

TEST_CASE("EllipCylinder tests")
{
    GIVEN("A shifted EllipCylinder on z axis")
    {
        index_t size = 10;
        IndexVector_t sizes(3);
        sizes << size, size, size;

        phantoms::Vec3i center;
        center << 5.0, 5.0, 5.0;

        phantoms::Vec2X<double> halfAxis;
        halfAxis << 2.0, 3.0;

        double length{3.};

        double amplit{1.};

        VolumeDescriptor dd(sizes);
        DataContainer<double> dc(dd);
        dc = 0;

        WHEN("Rasterize")
        {
            phantoms::EllipCylinder<double> ec{elsa::phantoms::Orientation::Z_AXIS, amplit, center,
                                               halfAxis, length};

            phantoms::rasterize<phantoms::Blending::ADDITION>(ec, dd, dc);

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
        index_t size = 20;
        IndexVector_t sizes(3);
        sizes << size, size, size;

        phantoms::Vec3i center;
        center << 7.0, 7.0, 7.0;

        phantoms::Vec2X<double> halfAxis;
        halfAxis << 1.0, 1.0;

        double length{5.};

        double amplit{1.};

        VolumeDescriptor dd(sizes);
        DataContainer<double> dc(dd);
        dc = 0;

        WHEN("Rasterize")
        {
            phantoms::EllipCylinder<double> ec{elsa::phantoms::Orientation::Y_AXIS, amplit, center,
                                               halfAxis, length};

            phantoms::rasterize<phantoms::Blending::ADDITION>(ec, dd, dc);

            IndexVector_t idx(3);

            WHEN("Check empty volume")
            {

                idx << 19, 19, 19;
                REQUIRE_EQ(dc(idx), 0);

                idx << 0, 0, 0;
                REQUIRE_EQ(dc(idx), 0);

                idx << 5, 5, 5;
                REQUIRE_EQ(dc(idx), 0);
            }
            WHEN("Check filled volume")
            {
                idx << 7, 7, 7;
                REQUIRE_EQ(dc(idx), amplit);
                idx << 7, 10, 7;
                REQUIRE_EQ(dc(idx), amplit);

                idx << 7, 5, 7;
                REQUIRE_EQ(dc(idx), amplit);
                idx << 6, 7, 7;
                REQUIRE_EQ(dc(idx), amplit);
                idx << 7, 7, 6;
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

        phantoms::Vec2X<double> halfAxis;
        halfAxis << 2.0, 3.0;

        double length{4.};

        double amplit{1.};

        VolumeDescriptor dd(sizes);
        DataContainer<double> dc(dd);
        dc = 0;

        WHEN("Rasterize")
        {
            phantoms::EllipCylinder<double> ec{elsa::phantoms::Orientation::X_AXIS, amplit, center,
                                               halfAxis, length};

            phantoms::rasterize<phantoms::Blending::ADDITION>(ec, dd, dc);

            IndexVector_t idx(3);

            WHEN("Check empty volume")
            {

                idx << 5, 10, 5;
                REQUIRE_EQ(dc(idx), 0);

                idx << 5, 5, 9;
                REQUIRE_EQ(dc(idx), 0);
            }
            WHEN("Check filled volume")
            {
                idx << 3, 5, 5;
                REQUIRE_EQ(dc(idx), amplit);
                idx << 7, 5, 5;
                REQUIRE_EQ(dc(idx), amplit);

                idx << 5, 5, 5;
                REQUIRE_EQ(dc(idx), amplit);
                idx << 5, 6, 5;
                REQUIRE_EQ(dc(idx), amplit);
                idx << 5, 7, 5;
                REQUIRE_EQ(dc(idx), amplit);
            }
        }
    }

    GIVEN("A shifted EllipCylinder on x axis with a length bigger than the x dimension of the data "
          "container")
    {
        index_t size = 10;
        IndexVector_t sizes(3);
        sizes << size, size, size;

        phantoms::Vec3i center;
        center << size / 2, size / 2, size / 2;

        phantoms::Vec2X<double> halfAxis;
        halfAxis << 2.0, 3.0;

        double length{100.};

        double amplit{1.};

        VolumeDescriptor dd(sizes);
        DataContainer<double> dc(dd);
        dc = 0;

        WHEN("Rasterize")
        {
            phantoms::EllipCylinder<double> ec{elsa::phantoms::Orientation::Z_AXIS, amplit, center,
                                               halfAxis, length};

            phantoms::rasterize<phantoms::Blending::ADDITION>(ec, dd, dc);

            IndexVector_t idx(3);

            WHEN("Check empty volume")
            {
                idx << 0, 5, 5;
                REQUIRE_EQ(dc(idx), 0);
                idx << 9, 5, 5;
                REQUIRE_EQ(dc(idx), 0);
            }
            WHEN("Check filled volume")
            {
                idx << 5, 5, 0;
                REQUIRE_EQ(dc(idx), amplit);
                idx << 5, 5, 5;
                REQUIRE_EQ(dc(idx), amplit);
                idx << 5, 5, 9;
                REQUIRE_EQ(dc(idx), amplit);
            }
        }
    }
}
