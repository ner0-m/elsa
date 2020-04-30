/**
 * \file test_RayGenerationBench.cpp
 *
 * \brief Benchmarks for ray generation
 *
 * \author David Frank
 */

#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <catch2/catch.hpp>
#include "VolumeDescriptor.h"
#include "Geometry.h"
#include <string>
#include <cstdlib>

using namespace elsa;
static const index_t dimension = 2;

void iterate2D(const Geometry& geo)
{
    for (real_t detPixel : std::initializer_list<real_t>{0.5, 2.5, 4.5}) {
        RealVector_t pixel(1);
        pixel << detPixel;
        BENCHMARK("Ray for detector at pixel " + std::to_string(detPixel))
        {
            return geo.computeRayTo(pixel);
        };
    }
}

void iterate3D(const Geometry& geo)
{
    for (real_t detPixel1 : std::initializer_list<real_t>{0.5, 2.5, 4.5}) {
        for (real_t detPixel2 : std::initializer_list<real_t>{0.5, 2.5, 4.5}) {
            RealVector_t pixel(2);
            pixel << detPixel1, detPixel2;
            BENCHMARK("Ray for detector at pixel " + std::to_string(detPixel1) + "/"
                      + std::to_string(detPixel2))
            {
                return geo.computeRayTo(pixel);
            };
        }
    }
}

TEST_CASE("Ray generation for 2D")
{
    IndexVector_t volCoeff(2);
    volCoeff << 5, 5;
    VolumeDescriptor ddVol(volCoeff);

    IndexVector_t detCoeff(1);
    detCoeff << 5;
    VolumeDescriptor ddDet(detCoeff);

    real_t s2c = 10;
    real_t c2d = 4;

    GIVEN("Geometry without offset and rotation")
    {
        Geometry g(s2c, c2d, 0, ddVol, ddDet);

        // test outer + central pixels
        iterate2D(g);
    }

    GIVEN("Geometry with offset but no rotation")
    {
        real_t offset = 2;
        Geometry g(s2c, c2d, 0, ddVol, ddDet, offset);

        // test outer + central pixels
        iterate2D(g);
    }

    GIVEN("Geometry at 90°, but no offset")
    {
        real_t angle = pi_t / 2; // 90 degrees
        Geometry g(s2c, c2d, angle, ddVol, ddDet);

        // test outer + central pixels
        iterate2D(g);
    }

    GIVEN("Geometry at 45° with offset")
    {
        real_t angle = pi_t / 4; // 45 degrees
        real_t cx = -1;
        real_t cy = 2;
        Geometry g(s2c, c2d, angle, ddVol, ddDet, 0, cx, cy);

        // test outer + central pixels
        iterate2D(g);
    }
}

TEST_CASE("Ray generation for 3D")
{
    IndexVector_t volCoeff(3);
    volCoeff << 5, 5, 5;
    VolumeDescriptor ddVol(volCoeff);

    IndexVector_t detCoeff(2);
    detCoeff << 5, 5;
    VolumeDescriptor ddDet(detCoeff);

    real_t s2c = 10;
    real_t c2d = 4;

    GIVEN("Geometry without offset and rotation")
    {
        Geometry g(s2c, c2d, ddVol, ddDet, 0);

        // test outer + central pixels
        iterate3D(g);
    }

    GIVEN("Geometry with offset but no rotation")
    {
        real_t px = -1;
        real_t py = 3;
        Geometry g(s2c, c2d, ddVol, ddDet, 0, 0, 0, px, py);

        // test outer + central pixels
        iterate3D(g);
    }

    GIVEN("Geometry at 90°, but no offset")
    {
        real_t angle = pi_t / 2;
        Geometry g(s2c, c2d, ddVol, ddDet, angle);

        // test outer + central pixels
        iterate3D(g);
    }

    GIVEN("Geometry at 45°/22.5 with offset")
    {
        real_t angle1 = pi_t / 4;
        real_t angle2 = pi_t / 2;
        RealVector_t offset(3);
        offset << 1, -2, -1;
        Geometry g(s2c, c2d, ddVol, ddDet, angle1, angle2, 0, 0, 0, offset[0], offset[1],
                   offset[2]);

        // test outer + central pixels
        iterate3D(g);
    }

    GIVEN("Geometry at 45/22.5/12.25 with offset")
    {
        real_t angle1 = pi_t / 4;
        real_t angle2 = pi_t / 2;
        real_t angle3 = pi_t / 8;

        RealMatrix_t rot1(3, 3);
        rot1 << std::cos(angle1), 0, std::sin(angle1), 0, 1, 0, -std::sin(angle1), 0,
            std::cos(angle1);

        RealMatrix_t rot2(3, 3);
        rot2 << std::cos(angle2), -std::sin(angle2), 0, std::sin(angle2), std::cos(angle2), 0, 0, 0,
            1;

        RealMatrix_t rot3(3, 3);
        rot3 << std::cos(angle3), 0, std::sin(angle3), 0, 1, 0, -std::sin(angle3), 0,
            std::cos(angle3);

        RealMatrix_t R = rot1 * rot2 * rot3;

        Geometry g(s2c, c2d, ddVol, ddDet, R);

        // test outer + central pixels
        iterate3D(g);
    }
}