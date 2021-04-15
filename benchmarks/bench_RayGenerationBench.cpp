/**
 * @file test_RayGenerationBench.cpp
 *
 * @brief Benchmarks for ray generation
 *
 * @author David Frank
 */

#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <catch2/catch.hpp>
#include "VolumeDescriptor.h"
#include "PlanarDetectorDescriptor.h"
#include "Geometry.h"
#include <string>
#include <cstdlib>

using namespace elsa;
using namespace elsa::geometry;

void iterate2D(const DetectorDescriptor& descriptor)
{
    for (auto detPixel : {0, 2, 4}) {
        IndexVector_t pixel(2);
        pixel << detPixel, 0; // 2nd dim is 0, as we access the 0th geometry pose
        BENCHMARK("Ray for detector at pixel " + std::to_string(detPixel))
        {
            return descriptor.computeRayFromDetectorCoord(pixel);
        };
    }
}

void iterate3D(const DetectorDescriptor& descriptor)
{
    for (auto detPixel1 : {0, 2, 4}) {
        for (auto detPixel2 : {0, 2, 4}) {
            IndexVector_t pixel(3);
            pixel << detPixel1, detPixel2, 0; // same as for 2d, last dim is for geometry access
            BENCHMARK("Ray for detector at pixel " + std::to_string(detPixel1) + "/"
                      + std::to_string(detPixel2))
            {
                return descriptor.computeRayFromDetectorCoord(pixel);
            };
        }
    }
}

TEST_CASE("Ray generation for 2D")
{
    IndexVector_t volCoeff(2);
    volCoeff << 5, 5;

    IndexVector_t detCoeff(2);
    detCoeff << 5, 1;

    real_t s2c = 10;
    real_t c2d = 4;

    auto volData = VolumeData2D{Size2D{volCoeff}};
    auto sinoData = SinogramData2D{Size2D{detCoeff}};

    GIVEN("Geometry without offset and rotation")
    {
        Geometry g(SourceToCenterOfRotation{s2c}, CenterOfRotationToDetector{c2d}, Degree{0},
                   std::move(volData), std::move(sinoData));
        PlanarDetectorDescriptor descriptor(detCoeff, {g});

        // test outer + central pixels
        iterate2D(descriptor);
    }

    GIVEN("Geometry with offset but no rotation")
    {
        Geometry g(SourceToCenterOfRotation{s2c}, CenterOfRotationToDetector{c2d}, Degree{0},
                   std::move(volData), std::move(sinoData), PrincipalPointOffset{2});
        PlanarDetectorDescriptor descriptor(detCoeff, {g});

        // test outer + central pixels
        iterate2D(descriptor);
    }

    GIVEN("Geometry at 90°, but no offset")
    {
        Geometry g(SourceToCenterOfRotation{s2c}, CenterOfRotationToDetector{c2d}, Degree{90},
                   std::move(volData), std::move(sinoData));
        PlanarDetectorDescriptor descriptor(detCoeff, {g});

        // test outer + central pixels
        iterate2D(descriptor);
    }

    GIVEN("Geometry at 45° with offset")
    {
        Geometry g(SourceToCenterOfRotation{s2c}, CenterOfRotationToDetector{c2d}, Degree{45},
                   std::move(volData), std::move(sinoData), PrincipalPointOffset{0},
                   RotationOffset2D{-1, 2});
        PlanarDetectorDescriptor descriptor(detCoeff, {g});

        // test outer + central pixels
        iterate2D(descriptor);
    }
}

TEST_CASE("Ray generation for 3D")
{
    IndexVector_t volCoeff(3);
    volCoeff << 5, 5, 5;
    VolumeDescriptor ddVol(volCoeff);

    IndexVector_t detCoeff(3);
    detCoeff << 5, 5, 1;
    VolumeDescriptor ddDet(detCoeff);

    real_t s2c = 10;
    real_t c2d = 4;

    auto volData = VolumeData3D{Size3D{volCoeff}};
    auto sinoData = SinogramData3D{Size3D{detCoeff}};

    GIVEN("Geometry without offset and rotation")
    {
        Geometry g(SourceToCenterOfRotation{s2c}, CenterOfRotationToDetector{c2d},
                   std::move(volData), std::move(sinoData), RotationAngles3D{Gamma(0)});
        PlanarDetectorDescriptor descriptor(detCoeff, {g});

        // test outer + central pixels
        iterate3D(descriptor);
    }

    GIVEN("Geometry with offset but no rotation")
    {
        Geometry g(SourceToCenterOfRotation{s2c}, CenterOfRotationToDetector{c2d},
                   std::move(volData), std::move(sinoData), RotationAngles3D{Gamma{0}},
                   PrincipalPointOffset2D{0, 0}, RotationOffset3D{-1, 3, 0});
        PlanarDetectorDescriptor descriptor(detCoeff, {g});

        // test outer + central pixels
        iterate3D(descriptor);
    }

    GIVEN("Geometry at 90°, but no offset")
    {
        Geometry g(SourceToCenterOfRotation{s2c}, CenterOfRotationToDetector{c2d},
                   std::move(volData), std::move(sinoData), RotationAngles3D{Gamma{pi_t / 2}});
        PlanarDetectorDescriptor descriptor(detCoeff, {g});

        // test outer + central pixels
        iterate3D(descriptor);
    }

    GIVEN("Geometry at 45°/22.5 with offset")
    {
        real_t angle1 = pi_t / 4;
        real_t angle2 = pi_t / 2;

        Geometry g(SourceToCenterOfRotation{s2c}, CenterOfRotationToDetector{c2d},
                   std::move(volData), std::move(sinoData),
                   RotationAngles3D{Gamma{angle1}, Beta{angle2}}, PrincipalPointOffset2D{0, 0},
                   RotationOffset3D{1, -2, -1});
        PlanarDetectorDescriptor descriptor(detCoeff, {g});

        // test outer + central pixels
        iterate3D(descriptor);
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

        PlanarDetectorDescriptor descriptor(detCoeff, {g});

        // test outer + central pixels
        iterate3D(descriptor);
    }
}
