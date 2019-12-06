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

using namespace elsa;

SCENARIO("Create a Circular Trajectory")
{
    index_t s = 64;

    GIVEN("A 2D descriptor and 256 angles")
    {
        index_t numberOfAngles = 256;
        IndexVector_t volSize(2);
        volSize << s, s;
        DataDescriptor desc{volSize};

        WHEN("We create a half circular trajectory for this scenario")
        {
            index_t halfCircular = 180;
            index_t diffCenterSource = s * 100;
            index_t diffCenterDetector = s;

            auto [geomList, sdesc] = CircleTrajectoryGenerator::createTrajectory(
                numberOfAngles, desc, halfCircular, diffCenterSource, diffCenterDetector);

            THEN("Every geomList in our list has the same camera center and the same projection "
                 "matrix")
            {
                const real_t sourceToCenter = diffCenterSource;
                const real_t centerToDetector = diffCenterDetector;

                real_t angle = (1.0 / (numberOfAngles - 1)) * halfCircular;
                for (int i = 0; i < numberOfAngles; ++i) {
                    real_t currAngle = i * angle * pi / 180.0;
                    Geometry tmpGeom(sourceToCenter, centerToDetector, currAngle, desc, *sdesc);

                    REQUIRE((tmpGeom.getCameraCenter() - geomList[i].getCameraCenter()).norm()
                            == Approx(0));
                    REQUIRE(
                        (tmpGeom.getProjectionMatrix() - geomList[i].getProjectionMatrix()).norm()
                        == Approx(0).margin(0.0000001));
                    REQUIRE((tmpGeom.getInverseProjectionMatrix()
                             - geomList[i].getInverseProjectionMatrix())
                                .norm()
                            == Approx(0).margin(0.0000001));
                }
            }
        }

        WHEN("We create a full circular trajectory for this scenario")
        {
            index_t halfCircular = 359;
            index_t diffCenterSource = s * 100;
            index_t diffCenterDetector = s;

            auto [geomList, sdesc] = CircleTrajectoryGenerator::createTrajectory(
                numberOfAngles, desc, halfCircular, diffCenterSource, diffCenterDetector);

            THEN("Every geomList in our list has the same camera center and the same projection "
                 "matrix")
            {
                const real_t sourceToCenter = diffCenterSource;
                const real_t centerToDetector = diffCenterDetector;

                real_t angle = (1.0 / (numberOfAngles - 1)) * halfCircular;
                for (int i = 0; i < numberOfAngles; ++i) {
                    real_t currAngle = i * angle * pi / 180.0;
                    Geometry tmpGeom(sourceToCenter, centerToDetector, currAngle, desc, *sdesc);

                    REQUIRE((tmpGeom.getCameraCenter() - geomList[i].getCameraCenter()).norm()
                            == Approx(0));
                    REQUIRE(
                        (tmpGeom.getProjectionMatrix() - geomList[i].getProjectionMatrix()).norm()
                        == Approx(0).margin(0.0000001));
                    REQUIRE((tmpGeom.getInverseProjectionMatrix()
                             - geomList[i].getInverseProjectionMatrix())
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
        DataDescriptor desc{volSize};

        WHEN("We create a half circular trajectory for this scenario")
        {
            index_t halfCircular = 180;
            index_t diffCenterSource = s * 100;
            index_t diffCenterDetector = s;

            auto [geomList, sdesc] = CircleTrajectoryGenerator::createTrajectory(
                numberOfAngles, desc, halfCircular, diffCenterSource, diffCenterDetector);

            THEN("Every geomList in our list has the same camera center and the same projection "
                 "matrix")
            {
                const real_t sourceToCenter = diffCenterSource;
                const real_t centerToDetector = diffCenterDetector;

                real_t angleInc = 1.0 * halfCircular / (numberOfAngles - 1);
                for (int i = 0; i < numberOfAngles; ++i) {
                    real_t angle = i * angleInc * pi / 180.0;
                    Geometry tmpGeom(sourceToCenter, centerToDetector, desc, *sdesc, angle);

                    REQUIRE((tmpGeom.getCameraCenter() - geomList[i].getCameraCenter()).norm()
                            == Approx(0));
                    REQUIRE(
                        (tmpGeom.getProjectionMatrix() - geomList[i].getProjectionMatrix()).norm()
                        == Approx(0).margin(0.0000001));
                    REQUIRE((tmpGeom.getInverseProjectionMatrix()
                             - geomList[i].getInverseProjectionMatrix())
                                .norm()
                            == Approx(0).margin(0.0000001));
                }
            }
        }

        WHEN("We create a full circular trajectory for this scenario")
        {
            index_t halfCircular = 359;
            index_t diffCenterSource = s * 100;
            index_t diffCenterDetector = s;

            auto [geomList, sdesc] = CircleTrajectoryGenerator::createTrajectory(
                numberOfAngles, desc, halfCircular, diffCenterSource, diffCenterDetector);

            THEN("Every geomList in our list has the same camera center and the same projection "
                 "matrix")
            {
                const real_t sourceToCenter = diffCenterSource;
                const real_t centerToDetector = diffCenterDetector;

                real_t angleInc = 1.0 * halfCircular / (numberOfAngles - 1);
                for (int i = 0; i < numberOfAngles; ++i) {
                    real_t angle = i * angleInc * pi / 180.0;
                    Geometry tmpGeom(sourceToCenter, centerToDetector, desc, *sdesc, angle);

                    REQUIRE((tmpGeom.getCameraCenter() - geomList[i].getCameraCenter()).norm()
                            == Approx(0));
                    REQUIRE(
                        (tmpGeom.getProjectionMatrix() - geomList[i].getProjectionMatrix()).norm()
                        == Approx(0).margin(0.0000001));
                    REQUIRE((tmpGeom.getInverseProjectionMatrix()
                             - geomList[i].getInverseProjectionMatrix())
                                .norm()
                            == Approx(0).margin(0.0000001));
                }
            }
        }
    }
}
