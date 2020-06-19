/**
 * \file test_Geometry.cpp
 *
 * \brief Test for Geometry class
 *
 * \author Tobias Lasser - initial code
 */

#include <catch2/catch.hpp>
#include "Geometry.h"
#include "VolumeDescriptor.h"

#include <iostream>

#include <Eigen/Geometry>

using namespace elsa;

SCENARIO("2D transformations")
{
    using namespace geometry;

    Eigen::IOFormat cleanFmt(4, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");

    IndexVector_t volCoeff(2);
    volCoeff << 1, 1;
    VolumeDescriptor ddVol(volCoeff);

    IndexVector_t detCoeff(2);
    detCoeff << 1, 1;
    VolumeDescriptor ddDet(detCoeff);

    GIVEN("Geometry setup with volume at center and source and detector (no rotatio)")
    {
        const auto s2c =
            SourceToCenterOfRotation{static_cast<real_t>(GENERATE(range(0, 10))) + .5f};
        const auto c2d =
            CenterOfRotationToDetector{static_cast<real_t>(GENERATE(range(0, 10))) + .5f};

        const auto angle = Degree{0};

        auto volData = VolumeData2D(Size2D{volCoeff});
        auto sinoData = SinogramData2D(Size2D{detCoeff});

        const Geometry g(s2c, c2d, angle, std::move(volData), std::move(sinoData));

        const auto source = g.getCameraCenter();
        const auto pp = g.getPrincipalPoint();

        const auto projMat = g.getProjectionMatrix();
        const auto invProjMat = g.getInverseProjectionMatrix();

        const auto rotMat = g.getRotationMatrix();
        const auto invRotMat = g.getInverseRotationMatrix();

        // clang-format off
        // This describes the geometry below
        //
        //        c     Source
        //
        //     ┌-----┐
        //     |     |
        //     |  x  |  Volume Center, at (0, 0)
        //     |     |
        //     └-----┘
        //
        //
        //     |--p--|  Detector, with principal point in the middle
        //
        // clang-format on

        INFO("distance from source to center of rotation: " << s2c.get());
        INFO("Distance from center of rotation to detector: " << c2d.get());

        THEN("Source is at the correct position")
        {
            const RealVector_t expected = (RealVector_t(2) << 0.5f, 0.5f - s2c.get()).finished();

            INFO("Source is at " << source.format(cleanFmt) << ", should be at "
                                 << expected.format(cleanFmt));

            CHECK(expected.isApprox(source, 0.0001f));
        }

        THEN("Principal point is at the correct position")
        {
            const RealVector_t expected = (RealVector_t(2) << 0.5f, 0.5f + c2d.get()).finished();

            INFO("Principal point is at " << pp.format(cleanFmt) << ", should be at "
                                          << expected.format(cleanFmt));

            CHECK(expected.isApprox(pp, 0.0001f));
        }

        WHEN("Transforming Source")
        {
            THEN("It's mapped correctly into local coordinate frame")
            {
                RealVector_t expected(2);
                expected << 0, 0;

                RealVector_t homSource(2);
                homSource << source;

                auto mapped = projMat * homSource.homogeneous();

                INFO("Source is at " << mapped.format(cleanFmt) << ", should be at "
                                     << expected.format(cleanFmt));

                CHECK(expected.isApprox(mapped, 0.0001f));
            }
        }

        WHEN("Transforming Prinipal point")
        {
            THEN("It's mapped correctly into local coordinate frame")
            {
                RealVector_t expected(1);
                expected << 0.5f;

                const auto mapped = (projMat * pp.homogeneous()).hnormalized();

                INFO("Projection matrix:\n" << projMat);
                INFO("Principal point (" << pp.format(cleanFmt) << ") is mapped to "
                                         << mapped.format(cleanFmt) << ", should be at "
                                         << expected.format(cleanFmt));

                CHECK(expected.isApprox(mapped, 0.0001f));
            }

            THEN("Inverse transformation gives direction from principal point to camera center")
            {
                const RealVector_t expected = (pp - source).normalized();

                RealVector_t localPP(1);
                localPP << 0.5f;

                const auto mapped = (invProjMat * localPP.homogeneous()).head(2).normalized();

                INFO("Direction form principal point to source is: "
                     << mapped.format(cleanFmt) << ", should be " << expected.format(cleanFmt));

                CHECK(expected.isApprox(mapped, 0.0001f));
            }
        }
    }
}

SCENARIO("Testing 2D geometries")
{
    using namespace geometry;
    GIVEN("some 2D setup")
    {
        IndexVector_t volCoeff(2);
        volCoeff << 5, 5;
        VolumeDescriptor ddVol(volCoeff);

        IndexVector_t detCoeff(2);
        detCoeff << 5, 1;
        VolumeDescriptor ddDet(detCoeff);

        real_t s2c = 10;
        real_t c2d = 4;

        WHEN("testing geometry without rotation/offsets")
        {
            Geometry g(SourceToCenterOfRotation{s2c}, CenterOfRotationToDetector{c2d}, Radian{0},
                       VolumeData2D{Size2D{volCoeff}}, SinogramData2D{Size2D{detCoeff}});

            THEN("a copy is the same")
            {
                Geometry gCopy(g);
                REQUIRE(gCopy == g);
            }

            THEN("then P and Pinv are inverse")
            {
                RealMatrix_t id(2, 2);
                id.setIdentity();

                auto P = g.getProjectionMatrix();
                auto Pinv = g.getInverseProjectionMatrix();
                RealMatrix_t result = (P * Pinv) - id;
                REQUIRE(result.sum() == Approx(0));
            }

            THEN("then the rotation matrix is correct")
            {
                RealMatrix_t id(2, 2);
                id.setIdentity();
                REQUIRE((g.getRotationMatrix() - id).sum() == Approx(0));
            }

            THEN("the camera center is correct")
            {
                auto c = g.getCameraCenter();
                auto o = ddVol.getLocationOfOrigin();
                REQUIRE((c[0] - o[0]) == Approx(0));
                REQUIRE((c[1] - o[1] + s2c) == Approx(0));
            }
        }

        WHEN("testing geometry without rotation but with principal point offset")
        {
            real_t px = 2;

            Geometry g(SourceToCenterOfRotation{s2c}, CenterOfRotationToDetector{c2d}, Radian{0},
                       VolumeData2D{Size2D{volCoeff}}, SinogramData2D{Size2D{detCoeff}}, {},
                       PrincipalPointOffset{px});

            THEN("a copy is the same")
            {
                Geometry gCopy(g);
                REQUIRE(gCopy == g);
            }

            THEN("then P and Pinv are inverse")
            {
                RealMatrix_t id(2, 2);
                id.setIdentity();

                auto P = g.getProjectionMatrix();
                auto Pinv = g.getInverseProjectionMatrix();
                RealMatrix_t result = (P * Pinv) - id;
                REQUIRE(result.sum() == Approx(0));
            }

            THEN("then the rotation matrix is correct")
            {
                RealMatrix_t id(2, 2);
                id.setIdentity();
                REQUIRE((g.getRotationMatrix() - id).sum() == Approx(0));
            }

            THEN("the camera center is correct")
            {
                auto c = g.getCameraCenter();
                auto o = ddVol.getLocationOfOrigin();
                REQUIRE((c[0] - o[0]) == Approx(0));
                REQUIRE((c[1] - o[1] + s2c) == Approx(0));
            }
        }

        WHEN("testing geometry with 90 degree rotation and no offsets")
        {
            Geometry g(SourceToCenterOfRotation{s2c}, CenterOfRotationToDetector{c2d}, Degree{90},
                       VolumeData2D{Size2D{volCoeff}}, SinogramData2D{Size2D{detCoeff}});

            THEN("a copy is the same")
            {
                Geometry gCopy(g);
                REQUIRE(gCopy == g);
            }

            THEN("then P and Pinv are inverse")
            {
                RealMatrix_t id(2, 2);
                id.setIdentity();

                auto P = g.getProjectionMatrix();
                auto Pinv = g.getInverseProjectionMatrix();
                RealMatrix_t result = (P * Pinv) - id;
                REQUIRE(result.sum() == Approx(0.0).margin(0.000001));
            }

            THEN("then the rotation matrix is correct")
            {
                RealMatrix_t rot(2, 2);
                rot << 0, -1, 1, 0;
                REQUIRE((g.getRotationMatrix() - rot).sum() == Approx(0).margin(0.0000001));
            }

            THEN("the camera center is correct")
            {
                auto c = g.getCameraCenter();
                auto o = ddVol.getLocationOfOrigin();

                REQUIRE((c[0] - o[0] + s2c) == Approx(0));
                REQUIRE((c[1] - o[1]) == Approx(0).margin(0.000001));
            }
        }

        WHEN("testing geometry with 45 degree rotation and offset center of rotation")
        {
            real_t angle = pi_t / 4; // 45 degrees
            real_t cx = -1;
            real_t cy = 2;

            Geometry g(SourceToCenterOfRotation{s2c}, CenterOfRotationToDetector{c2d},
                       Radian{angle}, VolumeData2D{Size2D{volCoeff}},
                       SinogramData2D{Size2D{detCoeff}}, {}, PrincipalPointOffset{0},
                       RotationOffset2D{cx, cy});

            THEN("a copy is the same")
            {
                Geometry gCopy(g);
                REQUIRE(gCopy == g);
            }

            THEN("then P and Pinv are inverse")
            {
                RealMatrix_t id(2, 2);
                id.setIdentity();

                auto P = g.getProjectionMatrix();
                auto Pinv = g.getInverseProjectionMatrix();
                RealMatrix_t result = (P * Pinv) - id;
                REQUIRE(result.sum() == Approx(0.0).margin(0.000001));
            }

            THEN("then the rotation matrix is correct")
            {
                RealMatrix_t rot(2, 2);
                rot << std::cos(angle), -std::sin(angle), std::sin(angle), std::cos(angle);
                REQUIRE((g.getRotationMatrix() - rot).sum() == Approx(0));
            }

            THEN("the camera center is correct")
            {
                auto c = g.getCameraCenter();
                auto o = ddVol.getLocationOfOrigin();

                real_t oldX = 0;
                real_t oldY = -s2c;
                real_t newX = std::cos(angle) * oldX + std::sin(angle) * oldY + o[0] + cx;
                real_t newY = -std::sin(angle) * oldX + std::cos(angle) * oldY + o[1] + cy;

                REQUIRE((c[0] - newX) == Approx(0).margin(0.000001));
                REQUIRE((c[1] - newY) == Approx(0).margin(0.000001));
            }
        }
    }
}

SCENARIO("Testing 3D geometries")
{
    using namespace geometry;
    GIVEN("some 3D setup")
    {
        IndexVector_t volCoeff(3);
        volCoeff << 5, 5, 5;
        VolumeDescriptor ddVol(volCoeff);

        IndexVector_t detCoeff(3);
        detCoeff << 5, 5, 1;
        VolumeDescriptor ddDet(detCoeff);

        real_t s2c = 10;
        real_t c2d = 4;

        WHEN("testing geometry without rotation/offsets")
        {
            Geometry g(SourceToCenterOfRotation{s2c}, CenterOfRotationToDetector{c2d},
                       VolumeData3D{Size3D{volCoeff}}, SinogramData3D{Size3D{detCoeff}},
                       RotationAngles3D{Radian{0}, Radian{0}, Radian{0}});

            THEN("a copy is the same")
            {
                Geometry gCopy(g);
                REQUIRE(gCopy == g);
            }

            THEN("then P and Pinv are inverse")
            {
                RealMatrix_t id(3, 3);
                id.setIdentity();

                auto P = g.getProjectionMatrix();
                auto Pinv = g.getInverseProjectionMatrix();
                RealMatrix_t result = (P * Pinv) - id;
                REQUIRE(result.sum() == Approx(0));
            }

            THEN("then the rotation matrix is correct")
            {
                RealMatrix_t id(3, 3);
                id.setIdentity();
                REQUIRE((g.getRotationMatrix() - id).sum() == Approx(0));
            }

            THEN("the camera center is correct")
            {
                auto c = g.getCameraCenter();
                auto o = ddVol.getLocationOfOrigin();
                REQUIRE((c[0] - o[0]) == Approx(0));
                REQUIRE((c[1] - o[1]) == Approx(0));
                REQUIRE((c[2] - o[2] + s2c) == Approx(0));
            }
        }

        WHEN("testing geometry without rotation but with principal point offsets")
        {
            real_t px = -1;
            real_t py = 3;

            Geometry g(SourceToCenterOfRotation{s2c}, CenterOfRotationToDetector{c2d},
                       VolumeData3D{Size3D{volCoeff}}, SinogramData3D{Size3D{detCoeff}},
                       RotationAngles3D{Radian{0}, Radian{0}, Radian{0}}, {},
                       PrincipalPointOffset2D{px, py});

            THEN("a copy is the same")
            {
                Geometry gCopy(g);
                REQUIRE(gCopy == g);
            }

            THEN("then P and Pinv are inverse")
            {
                RealMatrix_t id(3, 3);
                id.setIdentity();

                auto P = g.getProjectionMatrix();
                auto Pinv = g.getInverseProjectionMatrix();
                RealMatrix_t result = (P * Pinv) - id;
                REQUIRE(result.sum() == Approx(0));
            }

            THEN("then the rotation matrix is correct")
            {
                RealMatrix_t id(3, 3);
                id.setIdentity();
                REQUIRE((g.getRotationMatrix() - id).sum() == Approx(0));
            }

            THEN("the camera center is correct")
            {
                auto c = g.getCameraCenter();
                auto o = ddVol.getLocationOfOrigin();
                REQUIRE((c[0] - o[0]) == Approx(0));
                REQUIRE((c[1] - o[1]) == Approx(0));
                REQUIRE((c[2] - o[2] + s2c) == Approx(0));
            }
        }

        WHEN("testing geometry with 90 degree rotation and no offsets")
        {
            real_t angle = pi_t / 2;

            Geometry g(SourceToCenterOfRotation{s2c}, CenterOfRotationToDetector{c2d},
                       VolumeData3D{Size3D{volCoeff}}, SinogramData3D{Size3D{detCoeff}},
                       RotationAngles3D{Radian{angle}, Radian{0}, Radian{0}});

            THEN("a copy is the same")
            {
                Geometry gCopy(g);
                REQUIRE(gCopy == g);
            }

            THEN("then P and Pinv are inverse")
            {
                RealMatrix_t id(3, 3);
                id.setIdentity();

                auto P = g.getProjectionMatrix();
                auto Pinv = g.getInverseProjectionMatrix();
                RealMatrix_t result = (P * Pinv) - id;
                REQUIRE(result.sum() == Approx(0).margin(0.0000001));
            }

            THEN("then the rotation matrix is correct")
            {
                RealMatrix_t rot(3, 3);
                rot << std::cos(angle), 0, std::sin(angle), 0, 1, 0, -std::sin(angle), 0,
                    std::cos(angle);
                REQUIRE((g.getRotationMatrix() - rot).sum() == Approx(0));
            }

            THEN("the camera center is correct")
            {
                auto c = g.getCameraCenter();
                auto o = ddVol.getLocationOfOrigin();

                REQUIRE((c[0] - o[0] - s2c) == Approx(0));
                REQUIRE((c[1] - o[1]) == Approx(0).margin(0.000001));
                REQUIRE((c[2] - o[2]) == Approx(0).margin(0.000001));
            }
        }

        WHEN("testing geometry with 45/22.5 degree rotation and offset center of rotation")
        {
            real_t angle1 = pi_t / 4;
            real_t angle2 = pi_t / 2;
            RealVector_t offset(3);
            offset << 1, -2, -1;

            Geometry g(SourceToCenterOfRotation{s2c}, CenterOfRotationToDetector{c2d},
                       VolumeData3D{Size3D{volCoeff}}, SinogramData3D{Size3D{detCoeff}},
                       RotationAngles3D{Radian{angle1}, Radian{angle2}, Radian{0}}, {},
                       PrincipalPointOffset2D{0, 0}, RotationOffset3D{offset});

            THEN("a copy is the same")
            {
                Geometry gCopy(g);
                REQUIRE(gCopy == g);
            }

            THEN("then P and Pinv are inverse")
            {
                RealMatrix_t id(3, 3);
                id.setIdentity();

                auto P = g.getProjectionMatrix();
                auto Pinv = g.getInverseProjectionMatrix();
                RealMatrix_t result = (P * Pinv) - id;
                REQUIRE(result.sum() == Approx(0).margin(0.00001));
            }

            THEN("then the rotation matrix is correct")
            {
                RealMatrix_t rot1(3, 3);
                rot1 << std::cos(angle1), 0, std::sin(angle1), 0, 1, 0, -std::sin(angle1), 0,
                    std::cos(angle1);
                RealMatrix_t rot2(3, 3);
                rot2 << std::cos(angle2), -std::sin(angle2), 0, std::sin(angle2), std::cos(angle2),
                    0, 0, 0, 1;
                REQUIRE((g.getRotationMatrix() - rot1 * rot2).sum() == Approx(0));
            }

            THEN("the camera center is correct")
            {
                auto c = g.getCameraCenter();
                auto o = ddVol.getLocationOfOrigin();

                RealVector_t src(3);
                src[0] = 0;
                src[1] = 0;
                src[2] = -s2c;
                RealVector_t rotSrc = g.getRotationMatrix().transpose() * src + o + offset;

                REQUIRE((rotSrc - c).sum() == Approx(0).margin(0.000001));
            }
        }

        WHEN("testing geometry with 45/22.5/12.25 degree rotation as a rotation matrix")
        {
            real_t angle1 = pi_t / 4;
            real_t angle2 = pi_t / 2;
            real_t angle3 = pi_t / 8;
            RealMatrix_t rot1(3, 3);
            rot1 << std::cos(angle1), 0, std::sin(angle1), 0, 1, 0, -std::sin(angle1), 0,
                std::cos(angle1);
            RealMatrix_t rot2(3, 3);
            rot2 << std::cos(angle2), -std::sin(angle2), 0, std::sin(angle2), std::cos(angle2), 0,
                0, 0, 1;
            RealMatrix_t rot3(3, 3);
            rot3 << std::cos(angle3), 0, std::sin(angle3), 0, 1, 0, -std::sin(angle3), 0,
                std::cos(angle3);
            const RealMatrix_t R = rot1 * rot2 * rot3;

            Geometry g(s2c, c2d, VolumeData3D{Size3D{volCoeff}}, SinogramData3D{Size3D{detCoeff}},
                       R);

            THEN("a copy is the same")
            {
                Geometry gCopy(g);
                REQUIRE(gCopy == g);
            }

            THEN("then P and Pinv are inverse")
            {
                RealMatrix_t id(3, 3);
                id.setIdentity();

                auto P = g.getProjectionMatrix();
                auto Pinv = g.getInverseProjectionMatrix();
                RealMatrix_t result = (P * Pinv) - id;
                REQUIRE(result.sum() == Approx(0).margin(0.00001));
            }

            THEN("then the rotation matrix is correct")
            {
                REQUIRE((g.getRotationMatrix() - R).sum() == Approx(0));
            }

            THEN("the camera center is correct")
            {
                auto c = g.getCameraCenter();
                auto o = ddVol.getLocationOfOrigin();

                RealVector_t src(3);
                src[0] = 0;
                src[1] = 0;
                src[2] = -s2c;
                RealVector_t rotSrc = g.getRotationMatrix().transpose() * src + o;

                REQUIRE((rotSrc - c).sum() == Approx(0).margin(0.00001));
            }
        }
    }
}
