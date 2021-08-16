/**
 * @file test_Geometry.cpp
 *
 * @brief Test for Geometry class
 *
 * @author Tobias Lasser - initial code
 */

#include "doctest/doctest.h"
#include "Geometry.h"
#include "VolumeDescriptor.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("core");

TEST_CASE("Geometry: Testing 2D geometries")
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
                REQUIRE_EQ(gCopy, g);
            }

            THEN("then P and Pinv are inverse")
            {
                RealMatrix_t id(2, 2);
                id.setIdentity();

                auto P = g.getProjectionMatrix();
                auto Pinv = g.getInverseProjectionMatrix();
                RealMatrix_t result = (P * Pinv) - id;
                REQUIRE_EQ(result.sum(), Approx(0));
            }

            THEN("then the rotation matrix is correct")
            {
                RealMatrix_t id(2, 2);
                id.setIdentity();
                REQUIRE_EQ((g.getRotationMatrix() - id).sum(), Approx(0));
            }

            THEN("the camera center is correct")
            {
                auto c = g.getCameraCenter();
                auto o = ddVol.getLocationOfOrigin();
                REQUIRE_EQ((c[0] - o[0]), Approx(0));
                REQUIRE_EQ((c[1] - o[1] + s2c), Approx(0));
            }
        }

        WHEN("testing geometry without rotation but with principal point offset")
        {
            real_t px = 2;

            Geometry g(SourceToCenterOfRotation{s2c}, CenterOfRotationToDetector{c2d}, Radian{0},
                       VolumeData2D{Size2D{volCoeff}}, SinogramData2D{Size2D{detCoeff}},
                       PrincipalPointOffset{px});

            THEN("a copy is the same")
            {
                Geometry gCopy(g);
                REQUIRE_EQ(gCopy, g);
            }

            THEN("then P and Pinv are inverse")
            {
                RealMatrix_t id(2, 2);
                id.setIdentity();

                auto P = g.getProjectionMatrix();
                auto Pinv = g.getInverseProjectionMatrix();
                RealMatrix_t result = (P * Pinv) - id;
                REQUIRE_EQ(result.sum(), Approx(0));
            }

            THEN("then the rotation matrix is correct")
            {
                RealMatrix_t id(2, 2);
                id.setIdentity();
                REQUIRE_EQ((g.getRotationMatrix() - id).sum(), Approx(0));
            }

            THEN("the camera center is correct")
            {
                auto c = g.getCameraCenter();
                auto o = ddVol.getLocationOfOrigin();
                REQUIRE_EQ((c[0] - o[0]), Approx(0));
                REQUIRE_EQ((c[1] - o[1] + s2c), Approx(0));
            }
        }

        WHEN("testing geometry with 90 degree rotation and no offsets")
        {
            Geometry g(SourceToCenterOfRotation{s2c}, CenterOfRotationToDetector{c2d}, Degree{90},
                       VolumeData2D{Size2D{volCoeff}}, SinogramData2D{Size2D{detCoeff}});

            THEN("a copy is the same")
            {
                Geometry gCopy(g);
                REQUIRE_EQ(gCopy, g);
            }

            THEN("then P and Pinv are inverse")
            {
                RealMatrix_t id(2, 2);
                id.setIdentity();

                auto P = g.getProjectionMatrix();
                auto Pinv = g.getInverseProjectionMatrix();
                RealMatrix_t result = (P * Pinv) - id;
                REQUIRE_EQ(result.sum(), Approx(0.0).epsilon(0.01));
            }

            THEN("then the rotation matrix is correct")
            {
                RealMatrix_t rot(2, 2);
                rot << 0, -1, 1, 0;
                REQUIRE_EQ((g.getRotationMatrix() - rot).sum(), Approx(0).epsilon(0.01));
            }

            THEN("the camera center is correct")
            {
                auto c = g.getCameraCenter();
                auto o = ddVol.getLocationOfOrigin();

                REQUIRE_EQ((c[0] - o[0] + s2c), Approx(0));
                REQUIRE_EQ((c[1] - o[1]), Approx(0).epsilon(0.01));
            }
        }

        WHEN("testing geometry with 45 degree rotation and offset center of rotation")
        {
            real_t angle = pi_t / 4; // 45 degrees
            real_t cx = -1;
            real_t cy = 2;

            Geometry g(SourceToCenterOfRotation{s2c}, CenterOfRotationToDetector{c2d},
                       Radian{angle}, VolumeData2D{Size2D{volCoeff}},
                       SinogramData2D{Size2D{detCoeff}}, PrincipalPointOffset{0},
                       RotationOffset2D{cx, cy});

            THEN("a copy is the same")
            {
                Geometry gCopy(g);
                REQUIRE_EQ(gCopy, g);
            }

            THEN("then P and Pinv are inverse")
            {
                RealMatrix_t id(2, 2);
                id.setIdentity();

                auto P = g.getProjectionMatrix();
                auto Pinv = g.getInverseProjectionMatrix();
                RealMatrix_t result = (P * Pinv) - id;
                REQUIRE_EQ(result.sum(), Approx(0.0).epsilon(0.01));
            }

            THEN("then the rotation matrix is correct")
            {
                RealMatrix_t rot(2, 2);
                rot << std::cos(angle), -std::sin(angle), std::sin(angle), std::cos(angle);
                REQUIRE_EQ((g.getRotationMatrix() - rot).sum(), Approx(0));
            }

            THEN("the camera center is correct")
            {
                auto c = g.getCameraCenter();
                auto o = ddVol.getLocationOfOrigin();

                real_t oldX = 0;
                real_t oldY = -s2c;
                real_t newX = std::cos(angle) * oldX + std::sin(angle) * oldY + o[0] + cx;
                real_t newY = -std::sin(angle) * oldX + std::cos(angle) * oldY + o[1] + cy;

                REQUIRE_EQ((c[0] - newX), Approx(0).epsilon(0.01));
                REQUIRE_EQ((c[1] - newY), Approx(0).epsilon(0.01));
            }
        }
    }
}

TEST_CASE("Geometry: Testing 3D geometries")
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
                REQUIRE_EQ(gCopy, g);
            }

            THEN("then P and Pinv are inverse")
            {
                RealMatrix_t id(3, 3);
                id.setIdentity();

                auto P = g.getProjectionMatrix();
                auto Pinv = g.getInverseProjectionMatrix();
                RealMatrix_t result = (P * Pinv) - id;
                REQUIRE_EQ(result.sum(), Approx(0));
            }

            THEN("then the rotation matrix is correct")
            {
                RealMatrix_t id(3, 3);
                id.setIdentity();
                REQUIRE_EQ((g.getRotationMatrix() - id).sum(), Approx(0));
            }

            THEN("the camera center is correct")
            {
                auto c = g.getCameraCenter();
                auto o = ddVol.getLocationOfOrigin();
                REQUIRE_EQ((c[0] - o[0]), Approx(0));
                REQUIRE_EQ((c[1] - o[1]), Approx(0));
                REQUIRE_EQ((c[2] - o[2] + s2c), Approx(0));
            }
        }

        WHEN("testing geometry without rotation but with principal point offsets")
        {
            real_t px = -1;
            real_t py = 3;

            Geometry g(SourceToCenterOfRotation{s2c}, CenterOfRotationToDetector{c2d},
                       VolumeData3D{Size3D{volCoeff}}, SinogramData3D{Size3D{detCoeff}},
                       RotationAngles3D{Radian{0}, Radian{0}, Radian{0}},
                       PrincipalPointOffset2D{px, py});

            THEN("a copy is the same")
            {
                Geometry gCopy(g);
                REQUIRE_EQ(gCopy, g);
            }

            THEN("then P and Pinv are inverse")
            {
                RealMatrix_t id(3, 3);
                id.setIdentity();

                auto P = g.getProjectionMatrix();
                auto Pinv = g.getInverseProjectionMatrix();
                RealMatrix_t result = (P * Pinv) - id;
                REQUIRE_EQ(result.sum(), Approx(0));
            }

            THEN("then the rotation matrix is correct")
            {
                RealMatrix_t id(3, 3);
                id.setIdentity();
                REQUIRE_EQ((g.getRotationMatrix() - id).sum(), Approx(0));
            }

            THEN("the camera center is correct")
            {
                auto c = g.getCameraCenter();
                auto o = ddVol.getLocationOfOrigin();
                REQUIRE_EQ((c[0] - o[0]), Approx(0));
                REQUIRE_EQ((c[1] - o[1]), Approx(0));
                REQUIRE_EQ((c[2] - o[2] + s2c), Approx(0));
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
                REQUIRE_EQ(gCopy, g);
            }

            THEN("then P and Pinv are inverse")
            {
                RealMatrix_t id(3, 3);
                id.setIdentity();

                auto P = g.getProjectionMatrix();
                auto Pinv = g.getInverseProjectionMatrix();
                RealMatrix_t result = (P * Pinv) - id;
                REQUIRE_EQ(result.sum(), Approx(0).epsilon(0.01));
            }

            THEN("then the rotation matrix is correct")
            {
                RealMatrix_t rot(3, 3);
                rot << std::cos(angle), 0, std::sin(angle), 0, 1, 0, -std::sin(angle), 0,
                    std::cos(angle);
                REQUIRE_EQ((g.getRotationMatrix() - rot).sum(), Approx(0));
            }

            THEN("the camera center is correct")
            {
                auto c = g.getCameraCenter();
                auto o = ddVol.getLocationOfOrigin();

                REQUIRE_EQ((c[0] - o[0] - s2c), Approx(0));
                REQUIRE_EQ((c[1] - o[1]), Approx(0).epsilon(0.01));
                REQUIRE_EQ((c[2] - o[2]), Approx(0).epsilon(0.01));
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
                       RotationAngles3D{Radian{angle1}, Radian{angle2}, Radian{0}},
                       PrincipalPointOffset2D{0, 0}, RotationOffset3D{offset});

            THEN("a copy is the same")
            {
                Geometry gCopy(g);
                REQUIRE_EQ(gCopy, g);
            }

            THEN("then P and Pinv are inverse")
            {
                RealMatrix_t id(3, 3);
                id.setIdentity();

                auto P = g.getProjectionMatrix();
                auto Pinv = g.getInverseProjectionMatrix();
                RealMatrix_t result = (P * Pinv) - id;
                REQUIRE_EQ(result.sum(), Approx(0).epsilon(0.01));
            }

            THEN("then the rotation matrix is correct")
            {
                RealMatrix_t rot1(3, 3);
                rot1 << std::cos(angle1), 0, std::sin(angle1), 0, 1, 0, -std::sin(angle1), 0,
                    std::cos(angle1);
                RealMatrix_t rot2(3, 3);
                rot2 << std::cos(angle2), -std::sin(angle2), 0, std::sin(angle2), std::cos(angle2),
                    0, 0, 0, 1;
                REQUIRE_EQ((g.getRotationMatrix() - rot1 * rot2).sum(), Approx(0));
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

                REQUIRE_EQ((rotSrc - c).sum(), Approx(0).epsilon(0.01));
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
            RealMatrix_t R = rot1 * rot2 * rot3;

            Geometry g(s2c, c2d, ddVol, ddDet, R);

            THEN("a copy is the same")
            {
                Geometry gCopy(g);
                REQUIRE_EQ(gCopy, g);
            }

            THEN("then P and Pinv are inverse")
            {
                RealMatrix_t id(3, 3);
                id.setIdentity();

                auto P = g.getProjectionMatrix();
                auto Pinv = g.getInverseProjectionMatrix();
                RealMatrix_t result = (P * Pinv) - id;
                REQUIRE_EQ(result.sum(), Approx(0).epsilon(0.01));
            }

            THEN("then the rotation matrix is correct")
            {
                REQUIRE_EQ((g.getRotationMatrix() - R).sum(), Approx(0));
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

                REQUIRE_EQ((rotSrc - c).sum(), Approx(0).epsilon(0.01));
            }
        }
    }
}

TEST_SUITE_END();
