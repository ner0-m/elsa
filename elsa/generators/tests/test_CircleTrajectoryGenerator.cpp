/**
 * \file test_CircleTrajectoryGenerator.cpp
 *
 * \brief Test for CircleTrajectoryGenerator class
 *
 * \author David Frank - initial code
 * \author Nikola Dinev - fixes
 * \author Tobias Lasser - modernization, fixes
 */

#include <catch2/catch.hpp>

#include "CircleTrajectoryGenerator.h"
#include "Logger.h"
#include "VolumeDescriptor.h"

using namespace elsa;

SCENARIO("Create a Circular Trajectory")
{
    using namespace geometry;

    index_t s = 64;

    GIVEN("A 2D descriptor and 256 angles")
    {
        index_t numberOfAngles = 256;
        IndexVector_t volSize(2);
        volSize << s, s;
        VolumeDescriptor desc{volSize};

        WHEN("We create a half circular trajectory for this scenario")
        {
            index_t halfCircular = 180;
            auto diffCenterSource = static_cast<real_t>(s * 100);
            auto diffCenterDetector = static_cast<real_t>(s);

            auto sdesc = CircleTrajectoryGenerator::createTrajectory(
                numberOfAngles, desc, halfCircular, diffCenterSource, diffCenterDetector);

            THEN("Every geomList in our list has the same camera center and the same projection "
                 "matrix")
            {
                const real_t sourceToCenter = diffCenterSource;
                const real_t centerToDetector = diffCenterDetector;

                real_t angle = static_cast<real_t>(1.0) * static_cast<real_t>(halfCircular)
                               / real_t(numberOfAngles - 1);
                for (std::size_t i = 0; i < static_cast<std::size_t>(numberOfAngles); ++i) {
                    real_t currAngle = static_cast<real_t>(i) * angle * pi_t / 180.0f;
                    Geometry tmpGeom(SourceToCenterOfRotation{sourceToCenter},
                                     CenterOfRotationToDetector{centerToDetector},
                                     Radian{currAngle}, VolumeData2D{volSize},
                                     SinogramData2D{sdesc->getSpacingPerDimension(),
                                                    sdesc->getLocationOfOrigin()});

                    auto geom = sdesc->getGeometryAt(i);
                    CHECK(geom);

                    REQUIRE((tmpGeom.getCameraCenter() - geom->getCameraCenter()).norm()
                            == Approx(0));
                    REQUIRE((tmpGeom.getProjectionMatrix() - geom->getProjectionMatrix()).norm()
                            == Approx(0).margin(0.0000001));
                    REQUIRE(
                        (tmpGeom.getInverseProjectionMatrix() - geom->getInverseProjectionMatrix())
                            .norm()
                        == Approx(0).margin(0.0000001));
                    REQUIRE(
                        (tmpGeom.getInverseProjectionMatrix() - geom->getInverseProjectionMatrix())
                            .norm()
                        == Approx(0).margin(0.0000001));
                }
            }
        }

        WHEN("We create a full circular trajectory for this scenario")
        {
            index_t halfCircular = 359;
            auto diffCenterSource = static_cast<real_t>(s * 100);
            auto diffCenterDetector = static_cast<real_t>(s);

            auto sdesc = CircleTrajectoryGenerator::createTrajectory(
                numberOfAngles, desc, halfCircular, diffCenterSource, diffCenterDetector);

            THEN("Every geomList in our list has the same camera center and the same projection "
                 "matrix")
            {
                const real_t sourceToCenter = diffCenterSource;
                const real_t centerToDetector = diffCenterDetector;

                real_t angle = static_cast<real_t>(1.0) * static_cast<real_t>(halfCircular)
                               / static_cast<real_t>(numberOfAngles - 1);
                for (std::size_t i = 0; i < static_cast<std::size_t>(numberOfAngles); ++i) {
                    real_t currAngle = static_cast<real_t>(i) * angle * pi_t / 180.0f;

                    Geometry tmpGeom(SourceToCenterOfRotation{sourceToCenter},
                                     CenterOfRotationToDetector{centerToDetector},
                                     Radian{currAngle}, VolumeData2D{volSize},
                                     SinogramData2D{sdesc->getSpacingPerDimension(),
                                                    sdesc->getLocationOfOrigin()});

                    auto geom = sdesc->getGeometryAt(i);
                    CHECK(geom);

                    REQUIRE((tmpGeom.getCameraCenter() - geom->getCameraCenter()).norm()
                            == Approx(0));
                    REQUIRE((tmpGeom.getProjectionMatrix() - geom->getProjectionMatrix()).norm()
                            == Approx(0).margin(0.0000001));
                    REQUIRE(
                        (tmpGeom.getInverseProjectionMatrix() - geom->getInverseProjectionMatrix())
                            .norm()
                        == Approx(0).margin(0.0000001));
                    REQUIRE(
                        (tmpGeom.getInverseProjectionMatrix() - geom->getInverseProjectionMatrix())
                            .norm()
                        == Approx(0).margin(0.0000001));
                }
            }
        }
    }

    GIVEN("A 3D descriptor and 256 angles")
    {
        index_t numberOfAngles = 256;
        IndexVector_t volSize(3);
        volSize << s, s, s;
        VolumeDescriptor desc{volSize};

        WHEN("We create a half circular trajectory for this scenario")
        {
            index_t halfCircular = 180;
            auto diffCenterSource = static_cast<real_t>(s * 100);
            auto diffCenterDetector = static_cast<real_t>(s);

            auto sdesc = CircleTrajectoryGenerator::createTrajectory(
                numberOfAngles, desc, halfCircular, diffCenterSource, diffCenterDetector);

            THEN("Every geomList in our list has the same camera center and the same projection "
                 "matrix")
            {
                const real_t sourceToCenter = diffCenterSource;
                const real_t centerToDetector = diffCenterDetector;

                real_t angleInc = 1.0f * static_cast<real_t>(halfCircular)
                                  / static_cast<real_t>(numberOfAngles - 1);
                for (std::size_t i = 0; i < static_cast<std::size_t>(numberOfAngles); ++i) {
                    real_t angle = static_cast<real_t>(i) * angleInc * pi_t / 180.0f;

                    Geometry tmpGeom(SourceToCenterOfRotation{sourceToCenter},
                                     CenterOfRotationToDetector{centerToDetector},
                                     VolumeData3D{volSize},
                                     SinogramData3D{sdesc->getSpacingPerDimension(),
                                                    sdesc->getLocationOfOrigin()},
                                     RotationAngles3D{Gamma{angle}});

                    auto geom = sdesc->getGeometryAt(i);
                    CHECK(geom);

                    REQUIRE((tmpGeom.getCameraCenter() - geom->getCameraCenter()).norm()
                            == Approx(0));
                    REQUIRE((tmpGeom.getProjectionMatrix() - geom->getProjectionMatrix()).norm()
                            == Approx(0).margin(0.0000001));
                    REQUIRE(
                        (tmpGeom.getInverseProjectionMatrix() - geom->getInverseProjectionMatrix())
                            .norm()
                        == Approx(0).margin(0.0000001));
                    REQUIRE(
                        (tmpGeom.getInverseProjectionMatrix() - geom->getInverseProjectionMatrix())
                            .norm()
                        == Approx(0).margin(0.0000001));
                }
            }
        }

        WHEN("We create a full circular trajectory for this scenario")
        {
            const index_t halfCircular = 359;
            const auto diffCenterSource = static_cast<real_t>(s * 100);
            const auto diffCenterDetector = static_cast<real_t>(s);

            auto sdesc = CircleTrajectoryGenerator::createTrajectory(
                numberOfAngles, desc, halfCircular, diffCenterSource, diffCenterDetector);

            THEN("Every geomList in our list has the same camera center and the same projection "
                 "matrix")
            {
                const real_t sourceToCenter = diffCenterSource;
                const real_t centerToDetector = diffCenterDetector;

                real_t angleInc = 1.0f * static_cast<real_t>(halfCircular)
                                  / static_cast<real_t>(numberOfAngles - 1);
                for (std::size_t i = 0; i < static_cast<std::size_t>(numberOfAngles); ++i) {
                    real_t angle = static_cast<real_t>(i) * angleInc * pi_t / 180.0f;

                    Geometry tmpGeom(SourceToCenterOfRotation{sourceToCenter},
                                     CenterOfRotationToDetector{centerToDetector},
                                     VolumeData3D{volSize},
                                     SinogramData3D{sdesc->getSpacingPerDimension(),
                                                    sdesc->getLocationOfOrigin()},
                                     RotationAngles3D{Gamma{angle}});

                    auto geom = sdesc->getGeometryAt(i);
                    CHECK(geom);

                    REQUIRE((tmpGeom.getCameraCenter() - geom->getCameraCenter()).norm()
                            == Approx(0));
                    REQUIRE((tmpGeom.getProjectionMatrix() - geom->getProjectionMatrix()).norm()
                            == Approx(0).margin(0.0000001));
                    REQUIRE(
                        (tmpGeom.getInverseProjectionMatrix() - geom->getInverseProjectionMatrix())
                            .norm()
                        == Approx(0).margin(0.0000001));
                    REQUIRE(
                        (tmpGeom.getInverseProjectionMatrix() - geom->getInverseProjectionMatrix())
                            .norm()
                        == Approx(0).margin(0.0000001));
                }
            }
        }
    }
}
