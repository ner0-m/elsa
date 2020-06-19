/**
 * @file test_CurvedDetectorDescriptor.cpp
 *
 * @brief Test for CurvedDetectorDescriptor
 *
 * @author David Frank - initial code
 */

#include <catch2/catch.hpp>

#include "CurvedDetectorDescriptor.h"
#include "VolumeDescriptor.h"
#include "PlanarDetectorDescriptor.h"
#include "Logger.h"

#include <iostream>

using namespace elsa;
using namespace elsa::geometry;

using Ray = DetectorDescriptor::Ray;

SCENARIO("Testing 2D CurvedDetectorDescriptor")
{
    // Turn logger of
    Logger::setLevel(Logger::LogLevel::OFF);

    using namespace geometry;
    Eigen::IOFormat fmt(4, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");

    GIVEN("Given a 1x1 Volume and a single pose")
    {
        IndexVector_t volSize(2);
        volSize << 1, 1;
        VolumeDescriptor ddVol(volSize);

        // Test a couple of different number of detector pixels
        const auto numDetectorPixel = static_cast<index_t>(GENERATE(1, 3, 5, 7, 16, 32));

        IndexVector_t sinoSize(2);
        sinoSize << numDetectorPixel, 1;

        // Test a few different distances
        const auto s2c = GENERATE(0.5f, 100.5f);
        const auto c2d = GENERATE(0.5f, 100.5f);

        auto volData = VolumeData2D(Size2D{volSize});
        auto sinoData = SinogramData2D(Size2D{sinoSize});

        // Test with multiple (critical) angles
        const auto angle = static_cast<real_t>(GENERATE(0, 45, 90, 135, 180));

        Geometry g(SourceToCenterOfRotation{s2c}, CenterOfRotationToDetector{c2d}, Degree{angle},
                   std::move(volData), std::move(sinoData));

        CurvedDetectorDescriptor desc(sinoSize, {g}, 1.f);
        PlanarDetectorDescriptor planar(sinoSize, {g});

        auto rotMatrix = g.getInverseRotationMatrix();

        const auto fanAngle = g.getFanAngle();

        INFO("Detector size: " << numDetectorPixel);

        WHEN("Computing ray for each pixel in detector")
        {
            const auto pixelCoord =
                static_cast<real_t>(GENERATE_COPY(range<index_t>(0, numDetectorPixel))) + 0.5f;
            RealVector_t pixel(1);
            pixel << pixelCoord;

            INFO("Current pixel coord: " << pixelCoord << " / " << numDetectorPixel);

            const auto ray = desc.computeRayFromDetectorCoord(pixel, 0);
            const auto rayPlanar = planar.computeRayFromDetectorCoord(pixel, 0);

            THEN("Camera center is ray origin")
            {
                CHECK(g.getCameraCenter().isApprox(ray.origin()));
            }

            THEN("Ray direction is correct")
            {
                // map pixel coord to [0, 1], and then map this to [-fanAngle, fanAngle]
                // With property: if pixel == numDetectorPixel / 2 => rotAngle == 0
                const auto rotAngle =
                    Radian{((pixelCoord / static_cast<real_t>(numDetectorPixel)) * fanAngle * 2)
                           - fanAngle};

                // expected is [0, 1]^T rotated by correct amount
                RealVector_t expected(2);
                expected << std::sin(rotAngle), std::cos(rotAngle);
                expected.normalize();
                expected = rotMatrix * expected;

                INFO("Expected ray direction: " << expected.format(fmt));
                INFO("Ray direction: " << ray.direction().format(fmt));
                INFO("Planar detector ray direction: " << rayPlanar.direction().format(fmt));
                INFO("fan angle " << fanAngle.to_degree());

                CHECK(expected.isApprox(ray.direction()));
            }
        }

        WHEN("Computing rays to right edge of detector")
        {
            RealVector_t pixel(1);
            pixel << 0.f;

            auto ray = desc.computeRayFromDetectorCoord(pixel, 0);

            THEN("Camera center is ray origin")
            {
                // CHECK(g.getCameraCenter().isApprox(ray.origin()));
            }

            THEN("Ray direction is correct")
            {
                RealVector_t expected(2);
                expected << -1, 1;
                expected.normalize();
                expected = rotMatrix * expected;

                INFO("Expected ray direction: " << expected.format(fmt));
                INFO("Ray direction: " << ray.direction().format(fmt));

                // CHECK(expected.isApprox(ray.direction()));
            }
        }

        WHEN("Computing rays to left edge of detector")
        {
            RealVector_t pixel(1);
            pixel << static_cast<real_t>(numDetectorPixel);

            auto ray = desc.computeRayFromDetectorCoord(pixel, 0);

            THEN("Camera center is ray origin")
            {
                // CHECK(g.getCameraCenter().isApprox(ray.origin()));
            }

            THEN("Ray direction is correct")
            {
                RealVector_t expected(2);
                expected << 1, 1;
                expected.normalize();
                expected = rotMatrix * expected;

                INFO("Expected ray direction: " << expected.format(fmt));
                INFO("Ray direction: " << ray.direction().format(fmt));

                // CHECK(expected.isApprox(ray.direction()));
            }
        }
    }
}
