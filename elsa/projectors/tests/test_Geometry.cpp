/**
 * \file test_Geometry.cpp
 *
 * \brief Test for Geometry class
 *
 * \author Tobias Lasser - initial code
 */

#include <catch2/catch.hpp>
#include "Geometry.h"

#include <iostream>

using namespace elsa;

SCENARIO("Testing 2D geometries")
{
    GIVEN("some 2D setup")
    {
        IndexVector_t volCoeff(2);
        volCoeff << 5, 5;
        DataDescriptor ddVol(volCoeff);

        IndexVector_t detCoeff(1);
        detCoeff << 5;
        DataDescriptor ddDet(detCoeff);

        real_t s2c = 10;
        real_t c2d = 4;

        WHEN("testing geometry without rotation/offsets")
        {
            Geometry g(s2c, c2d, 0, ddVol, ddDet);

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

            THEN("rays make sense")
            {
                // test outer + central pixels
                for (real_t detPixel : std::initializer_list<real_t>{0.5, 2.5, 4.5}) {
                    RealVector_t pixel(1);
                    pixel << detPixel;
                    auto [ro, rd] = g.computeRayTo(pixel);

                    auto c = g.getCameraCenter();
                    REQUIRE((ro - c).sum() == Approx(0));

                    real_t factor =
                        (std::abs(rd[0]) > 0) ? ((pixel[0] - ro[0]) / rd[0]) : (s2c + c2d);
                    real_t detCoordY = ro[1] + factor * rd[1];
                    REQUIRE(detCoordY == Approx(ddVol.getLocationOfOrigin()[1] + c2d));
                }
            }
        }

        WHEN("testing geometry without rotation but with principal point offset")
        {
            real_t px = 2;
            Geometry g(s2c, c2d, 0, ddVol, ddDet, px);

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

            THEN("rays make sense")
            {
                // test outer + central pixels
                for (real_t detPixel : std::initializer_list<real_t>{0.5, 2.5, 4.5}) {
                    RealVector_t pixel(1);
                    pixel << detPixel;
                    auto [ro, rd] = g.computeRayTo(pixel);

                    auto c = g.getCameraCenter();
                    REQUIRE((ro - c).sum() == Approx(0));

                    real_t factor =
                        (std::abs(rd[0]) > 0) ? ((pixel[0] - ro[0] - px) / rd[0]) : (s2c + c2d);
                    real_t detCoordY = ro[1] + factor * rd[1];
                    REQUIRE(detCoordY == Approx(ddVol.getLocationOfOrigin()[1] + c2d));
                }
            }
        }

        WHEN("testing geometry with 90 degree rotation and no offsets")
        {
            real_t angle = pi_t / 2; // 90 degrees
            Geometry g(s2c, c2d, angle, ddVol, ddDet);

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

            THEN("rays make sense")
            {
                // test outer + central pixels
                for (real_t detPixel : std::initializer_list<real_t>{0.5, 2.5, 4.5}) {
                    RealVector_t pixel(1);
                    pixel << detPixel;
                    auto [ro, rd] = g.computeRayTo(pixel);

                    auto c = g.getCameraCenter();
                    REQUIRE((ro - c).sum() == Approx(0));

                    real_t factor =
                        (std::abs(rd[1]) > 0.0000001) ? ((ro[1] - pixel[0]) / rd[1]) : (s2c + c2d);
                    real_t detCoordX = ro[0] + factor * rd[0];
                    REQUIRE(detCoordX == Approx(ddVol.getLocationOfOrigin()[0] + c2d));
                }
            }
        }

        WHEN("testing geometry with 45 degree rotation and offset center of rotation")
        {
            real_t angle = pi_t / 4; // 45 degrees
            real_t cx = -1;
            real_t cy = 2;
            Geometry g(s2c, c2d, angle, ddVol, ddDet, 0, cx, cy);

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

            THEN("rays make sense")
            {
                // test outer + central pixels
                for (real_t detPixel : std::initializer_list<real_t>{0.5, 2.5, 4.5}) {
                    RealVector_t pixel(1);
                    pixel << detPixel;
                    auto [ro, rd] = g.computeRayTo(pixel);

                    auto c = g.getCameraCenter();
                    REQUIRE((ro - c).sum() == Approx(0.0));

                    auto o = ddVol.getLocationOfOrigin();
                    RealVector_t detCoordWorld(2);
                    detCoordWorld << detPixel - o[0], c2d;
                    RealVector_t rotD = g.getRotationMatrix().transpose() * detCoordWorld;
                    rotD[0] += o[0] + cx;
                    rotD[1] += o[1] + cy;

                    real_t factor = (rotD[0] - ro[0]) / rd[0]; // rd[0] won't be 0 here!
                    real_t detCoord = ro[1] + factor * rd[1];
                    REQUIRE(detCoord == Approx(rotD[1]));
                }
            }
        }
    }
}

SCENARIO("Testing 3D geometries")
{
    GIVEN("some 3D setup")
    {
        IndexVector_t volCoeff(3);
        volCoeff << 5, 5, 5;
        DataDescriptor ddVol(volCoeff);

        IndexVector_t detCoeff(2);
        detCoeff << 5, 5;
        DataDescriptor ddDet(detCoeff);

        real_t s2c = 10;
        real_t c2d = 4;

        WHEN("testing geometry without rotation/offsets")
        {
            Geometry g(s2c, c2d, ddVol, ddDet, 0);

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

            THEN("rays make sense")
            {
                // test outer + central pixels
                for (real_t detPixel1 : std::initializer_list<real_t>{0.5, 2.5, 4.5}) {
                    for (real_t detPixel2 : std::initializer_list<real_t>{0.5, 2.5, 4.5}) {
                        RealVector_t pixel(2);
                        pixel << detPixel1, detPixel2;
                        auto [ro, rd] = g.computeRayTo(pixel);

                        auto c = g.getCameraCenter();
                        REQUIRE((ro - c).sum() == Approx(0));

                        auto o = ddVol.getLocationOfOrigin();
                        RealVector_t detCoordWorld(3);
                        detCoordWorld << detPixel1 - o[0], detPixel2 - o[1], c2d;
                        RealVector_t rotD = g.getRotationMatrix().transpose() * detCoordWorld + o;

                        real_t factor = 0;
                        if (std::abs(rd[0]) > 0)
                            factor = (rotD[0] - ro[0]) / rd[0];
                        else if (std::abs(rd[1]) > 0)
                            factor = (rotD[1] - ro[1]) / rd[1];
                        else if (std::abs(rd[2]) > 0)
                            factor = (rotD[2] - ro[2] / rd[2]);
                        REQUIRE((ro[0] + factor * rd[0]) == Approx(rotD[0]));
                        REQUIRE((ro[1] + factor * rd[1]) == Approx(rotD[1]));
                        REQUIRE((ro[2] + factor * rd[2]) == Approx(rotD[2]));
                    }
                }
            }
        }

        WHEN("testing geometry without rotation but with principal point offsets")
        {
            real_t px = -1;
            real_t py = 3;
            Geometry g(s2c, c2d, ddVol, ddDet, 0, 0, 0, px, py);

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

            THEN("rays make sense")
            {
                // test outer + central pixels
                for (real_t detPixel1 : std::initializer_list<real_t>{0.5, 2.5, 4.5}) {
                    for (real_t detPixel2 : std::initializer_list<real_t>{0.5, 2.5, 4.5}) {
                        RealVector_t pixel(2);
                        pixel << detPixel1, detPixel2;
                        auto [ro, rd] = g.computeRayTo(pixel);

                        auto c = g.getCameraCenter();
                        REQUIRE((ro - c).sum() == Approx(0));

                        auto o = ddVol.getLocationOfOrigin();
                        RealVector_t detCoordWorld(3);
                        detCoordWorld << detPixel1 - o[0] - px, detPixel2 - o[1] - py, c2d;
                        RealVector_t rotD = g.getRotationMatrix().transpose() * detCoordWorld + o;

                        real_t factor = 0;
                        if (std::abs(rd[0]) > 0)
                            factor = (rotD[0] - ro[0]) / rd[0];
                        else if (std::abs(rd[1]) > 0)
                            factor = (rotD[1] - ro[1]) / rd[1];
                        else if (std::abs(rd[2]) > 0)
                            factor = (rotD[2] - ro[2] / rd[2]);
                        REQUIRE((ro[0] + factor * rd[0]) == Approx(rotD[0]));
                        REQUIRE((ro[1] + factor * rd[1]) == Approx(rotD[1]));
                        REQUIRE((ro[2] + factor * rd[2]) == Approx(rotD[2]));
                    }
                }
            }
        }

        WHEN("testing geometry with 90 degree rotation and no offsets")
        {
            real_t angle = pi_t / 2;
            Geometry g(s2c, c2d, ddVol, ddDet, angle);

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

            THEN("rays make sense")
            {
                // test outer + central pixels
                for (real_t detPixel1 : std::initializer_list<real_t>{0.5, 2.5, 4.5}) {
                    for (real_t detPixel2 : std::initializer_list<real_t>{0.5, 2.5, 4.5}) {
                        RealVector_t pixel(2);
                        pixel << detPixel1, detPixel2;
                        auto [ro, rd] = g.computeRayTo(pixel);

                        auto c = g.getCameraCenter();
                        REQUIRE((ro - c).sum() == Approx(0));

                        auto o = ddVol.getLocationOfOrigin();
                        RealVector_t detCoordWorld(3);
                        detCoordWorld << detPixel1 - o[0], detPixel2 - o[1], c2d;
                        RealVector_t rotD = g.getRotationMatrix().transpose() * detCoordWorld + o;

                        real_t factor = 0;
                        if (std::abs(rd[0]) > 0)
                            factor = (rotD[0] - ro[0]) / rd[0];
                        else if (std::abs(rd[1]) > 0)
                            factor = (rotD[1] - ro[1]) / rd[1];
                        else if (std::abs(rd[2]) > 0)
                            factor = (rotD[2] - ro[2] / rd[2]);
                        REQUIRE((ro[0] + factor * rd[0]) == Approx(rotD[0]));
                        REQUIRE((ro[1] + factor * rd[1]) == Approx(rotD[1]));
                        REQUIRE((ro[2] + factor * rd[2]) == Approx(rotD[2]));
                    }
                }
            }
        }

        WHEN("testing geometry with 45/22.5 degree rotation and offset center of rotation")
        {
            real_t angle1 = pi_t / 4;
            real_t angle2 = pi_t / 2;
            RealVector_t offset(3);
            offset << 1, -2, -1;
            Geometry g(s2c, c2d, ddVol, ddDet, angle1, angle2, 0, 0, 0, offset[0], offset[1],
                       offset[2]);

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

            THEN("rays make sense")
            {
                // test outer + central pixels
                for (real_t detPixel1 : std::initializer_list<real_t>{0.5, 2.5, 4.5}) {
                    for (real_t detPixel2 : std::initializer_list<real_t>{0.5, 2.5, 4.5}) {
                        RealVector_t pixel(2);
                        pixel << detPixel1, detPixel2;
                        auto [ro, rd] = g.computeRayTo(pixel);

                        auto c = g.getCameraCenter();
                        REQUIRE((ro - c).sum() == Approx(0));

                        auto o = ddVol.getLocationOfOrigin();
                        RealVector_t detCoordWorld(3);
                        detCoordWorld << detPixel1 - o[0], detPixel2 - o[1], c2d;
                        RealVector_t rotD =
                            g.getRotationMatrix().transpose() * detCoordWorld + o + offset;

                        real_t factor = 0;
                        if (std::abs(rd[0]) > 0.000001)
                            factor = (rotD[0] - ro[0]) / rd[0];
                        else if (std::abs(rd[1]) > 0.000001)
                            factor = (rotD[1] - ro[1]) / rd[1];
                        else if (std::abs(rd[2]) > 0.000001)
                            factor = (rotD[2] - ro[2] / rd[2]);
                        REQUIRE((ro[0] + factor * rd[0]) == Approx(rotD[0]));
                        REQUIRE((ro[1] + factor * rd[1]) == Approx(rotD[1]));
                        REQUIRE((ro[2] + factor * rd[2]) == Approx(rotD[2]));
                    }
                }
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

            THEN("rays make sense")
            {
                // test outer + central pixels
                for (real_t detPixel1 : std::initializer_list<real_t>{0.5, 2.5, 4.5}) {
                    for (real_t detPixel2 : std::initializer_list<real_t>{0.5, 2.5, 4.5}) {
                        RealVector_t pixel(2);
                        pixel << detPixel1, detPixel2;
                        auto [ro, rd] = g.computeRayTo(pixel);

                        auto c = g.getCameraCenter();
                        REQUIRE((ro - c).sum() == Approx(0));

                        auto o = ddVol.getLocationOfOrigin();
                        RealVector_t detCoordWorld(3);
                        detCoordWorld << detPixel1 - o[0], detPixel2 - o[1], c2d;
                        RealVector_t rotD = g.getRotationMatrix().transpose() * detCoordWorld + o;

                        real_t factor = 0;
                        if (std::abs(rd[0]) > 0.000001)
                            factor = (rotD[0] - ro[0]) / rd[0];
                        else if (std::abs(rd[1]) > 0.000001)
                            factor = (rotD[1] - ro[1]) / rd[1];
                        else if (std::abs(rd[2]) > 0.000001)
                            factor = (rotD[2] - ro[2] / rd[2]);
                        REQUIRE((ro[0] + factor * rd[0]) == Approx(rotD[0]));
                        REQUIRE((ro[1] + factor * rd[1]) == Approx(rotD[1]));
                        REQUIRE((ro[2] + factor * rd[2]) == Approx(rotD[2]));
                    }
                }
            }
        }
        
        WHEN("testing geometry from projection matrix with 45/22.5/12.25 degree rotation") {
            real_t angle1 = pi / 4;
            real_t angle2 = pi / 2;
            real_t angle3 = pi / 8;
            RealMatrix_t rot1(3, 3);
            rot1 << std::cos(angle1), 0, std::sin(angle1), 0, 1, 0, -std::sin(angle1), 0, std::cos(angle1);
            RealMatrix_t rot2(3, 3);
            rot2 << std::cos(angle2), -std::sin(angle2), 0, std::sin(angle2), std::cos(angle2), 0, 0, 0, 1;
            RealMatrix_t rot3(3, 3);
            rot3 << std::cos(angle3), 0, std::sin(angle3), 0, 1, 0, -std::sin(angle3), 0, std::cos(angle3);
            RealMatrix_t R = rot1 * rot2 * rot3;

            Geometry g(s2c, c2d, ddVol, ddDet, R);
            
            Geometry g1(s2c, c2d, g.getProjectionMatrix(), ddVol, ddDet);
            
            THEN("a copy is the same") {
                Geometry gCopy(g1);
                REQUIRE(gCopy == g1);
            }
            THEN("then the projection matrices of both geometries are the same") {
                auto P = g.getProjectionMatrix();
                auto P1 = g1.getProjectionMatrix();
                RealMatrix_t result = (P1 - P);
                REQUIRE(result.sum() == Approx(0).margin(0.000001));
            }
            THEN("then the  inv projection matrices of both geometries are the same") {
                auto P = g.getInverseProjectionMatrix();
                auto P1 = g1.getInverseProjectionMatrix();
                RealMatrix_t result = (P1 - P);
                REQUIRE(result.sum() == Approx(0).margin(0.000001));
            }
            THEN("then the roation matrices of both geometries are the same") {
                auto R = g.getRotationMatrix();
                auto R1 = g1.getRotationMatrix();
                RealMatrix_t result = (R1 - R);
                REQUIRE(result.sum() == Approx(0).margin(0.000001));
            }
            THEN("then the  camera centers of both geometries are the same") {
                auto C = g.getCameraCenter();
                auto C1 = g1.getCameraCenter();
                RealVector_t result = (C - C1);
                REQUIRE(result.sum() == Approx(0));
            }
        }
        
        WHEN("testing geometry from projection matrix without rotation but with principal point offsets") {
            real_t px = -1;
            real_t py = 3;
            Geometry g(s2c, c2d, ddVol, ddDet, 0, 0, 0, px, py);
            
            Geometry g1(s2c, c2d, g.getProjectionMatrix(), ddVol, ddDet,px,py);
            
            THEN("a copy is the same") {
                Geometry gCopy(g1);
                REQUIRE(gCopy == g1);
            }
            THEN("then the projection matrices of both geometries are the same") {
                auto P = g.getProjectionMatrix();
                auto P1 = g1.getProjectionMatrix();
                RealMatrix_t result = (P1 - P);
                REQUIRE(result.sum() == Approx(0));
            }
            THEN("then the  inv projection matrices of both geometries are the same") {
                auto P = g.getInverseProjectionMatrix();
                auto P1 = g1.getInverseProjectionMatrix();
                RealMatrix_t result = (P1 - P);
                REQUIRE(result.sum() == Approx(0).margin(0.000001));
            }
            THEN("then the roation matrices of both geometries are the same") {
                auto R = g.getRotationMatrix();
                auto R1 = g1.getRotationMatrix();
                RealMatrix_t result = (R1 - R);
                REQUIRE(result.sum() == Approx(0).margin(0.000001));
            }
            THEN("then the  camera centers of both geometries are the same") {
                auto C = g.getCameraCenter();
                auto C1 = g1.getCameraCenter();
                RealVector_t result = (C1 - C);
                REQUIRE(result.sum() == Approx(0));
                REQUIRE(C1 == C);
            }
        }    
    }
}