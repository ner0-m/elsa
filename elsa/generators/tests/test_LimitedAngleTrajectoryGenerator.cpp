/**
 * @file test_LimitedAngleTrajectoryGenerator.cpp
 *
 * @brief Tests for the LimitedAngleTrajectoryGenerator class
 *
 * @author Andi Braimllari
 */

#include "LimitedAngleTrajectoryGenerator.h"
#include "Logger.h"
#include "VolumeDescriptor.h"

#include "testHelpers.h"
#include "doctest/doctest.h"

using namespace elsa;
using namespace doctest;

TEST_CASE("LimitedAngleTrajectoryGenerator: Create a mirrored Limited Angle Trajectory")
{
    using namespace geometry;

    const index_t s = 64;

    // Detector size is the volume size scaled by the square root of 2
    const auto expectedDetectorSize = static_cast<index_t>(s * std::sqrt(2));

    GIVEN("A 2D descriptor and 256 angles")
    {
        index_t numberOfAngles = 256;
        IndexVector_t volSize(2);
        volSize << s, s;
        VolumeDescriptor desc{volSize};

        WHEN("We create a half circular limited angle trajectory for this scenario")
        {
            index_t halfCircular = 180;
            real_t diffCenterSource{s * 100};
            real_t diffCenterDetector{s};
            std::pair missingWedgeAngles(geometry::Degree(110), geometry::Degree(170));

            auto sinoDescriptor = LimitedAngleTrajectoryGenerator::createTrajectory(
                numberOfAngles, missingWedgeAngles, desc, halfCircular, diffCenterSource,
                diffCenterDetector);

            // check that the detector size is correct
            REQUIRE_EQ(sinoDescriptor->getNumberOfCoefficientsPerDimension()[0],
                       expectedDetectorSize);

            THEN("Every geomList in our list has the same camera center and the same projection "
                 "matrix")
            {
                const real_t sourceToCenter = diffCenterSource;
                const real_t centerToDetector = diffCenterDetector;

                real_t wedgeArc = 2 * (missingWedgeAngles.second - missingWedgeAngles.first);

                const real_t angleIncrement = (static_cast<real_t>(halfCircular) - wedgeArc)
                                              / (static_cast<real_t>(numberOfAngles));

                index_t j = 0;
                for (index_t i = 0;; ++i) {
                    Radian angle = Degree{static_cast<real_t>(i) * angleIncrement};

                    if (angle.to_degree() >= static_cast<real_t>(halfCircular)) {
                        break;
                    }

                    if (!((angle.to_degree() >= missingWedgeAngles.first
                           && angle.to_degree() <= missingWedgeAngles.second)
                          || (angle.to_degree() >= (missingWedgeAngles.first + 180)
                              && angle.to_degree() <= (missingWedgeAngles.second + 180)))) {
                        Geometry tmpGeom(SourceToCenterOfRotation{sourceToCenter},
                                         CenterOfRotationToDetector{centerToDetector},
                                         Radian{angle}, VolumeData2D{volSize},
                                         SinogramData2D{sinoDescriptor->getSpacingPerDimension(),
                                                        sinoDescriptor->getLocationOfOrigin()});

                        auto geom = sinoDescriptor->getGeometryAt(j++);
                        CHECK(geom);

                        const auto centerNorm =
                            (tmpGeom.getCameraCenter() - geom->getCameraCenter()).norm();
                        const auto projMatNorm =
                            (tmpGeom.getProjectionMatrix() - geom->getProjectionMatrix()).norm();
                        const auto invProjMatNorm = (tmpGeom.getInverseProjectionMatrix()
                                                     - geom->getInverseProjectionMatrix())
                                                        .norm();
                        REQUIRE_UNARY(checkApproxEq(centerNorm, 0));
                        REQUIRE_UNARY(checkApproxEq(projMatNorm, 0, 0.0000001));
                        REQUIRE_UNARY(checkApproxEq(invProjMatNorm, 0, 0.0000001));
                    }
                }
            }
        }

        WHEN("We create a full circular limited angle trajectory for this scenario")
        {
            index_t fullyCircular = 359;
            real_t diffCenterSource{s * 100};
            real_t diffCenterDetector{s};
            std::pair missingWedgeAngles(geometry::Degree(30), geometry::Degree(70));

            auto sinoDescriptor = LimitedAngleTrajectoryGenerator::createTrajectory(
                numberOfAngles, missingWedgeAngles, desc, fullyCircular, diffCenterSource,
                diffCenterDetector);

            // check that the detector size is correct
            REQUIRE_EQ(sinoDescriptor->getNumberOfCoefficientsPerDimension()[0],
                       expectedDetectorSize);

            THEN("Every geomList in our list has the same camera center and the same projection "
                 "matrix")
            {
                const real_t sourceToCenter = diffCenterSource;
                const real_t centerToDetector = diffCenterDetector;

                real_t wedgeArc = 2 * (missingWedgeAngles.second - missingWedgeAngles.first);

                const real_t angleIncrement = (static_cast<real_t>(fullyCircular) - wedgeArc)
                                              / (static_cast<real_t>(numberOfAngles));

                index_t j = 0;
                for (index_t i = 0;; ++i) {
                    Radian angle = Degree{static_cast<real_t>(i) * angleIncrement};

                    if (angle.to_degree() > static_cast<real_t>(fullyCircular)) {
                        break;
                    }

                    if (!((angle.to_degree() >= missingWedgeAngles.first
                           && angle.to_degree() <= missingWedgeAngles.second)
                          || (angle.to_degree() >= (missingWedgeAngles.first + 180)
                              && angle.to_degree() <= (missingWedgeAngles.second + 180)))) {
                        Geometry tmpGeom(SourceToCenterOfRotation{sourceToCenter},
                                         CenterOfRotationToDetector{centerToDetector},
                                         Radian{angle}, VolumeData2D{volSize},
                                         SinogramData2D{sinoDescriptor->getSpacingPerDimension(),
                                                        sinoDescriptor->getLocationOfOrigin()});

                        auto geom = sinoDescriptor->getGeometryAt(j++);
                        CHECK(geom);

                        const auto centerNorm =
                            (tmpGeom.getCameraCenter() - geom->getCameraCenter()).norm();
                        const auto projMatNorm =
                            (tmpGeom.getProjectionMatrix() - geom->getProjectionMatrix()).norm();
                        const auto invProjMatNorm = (tmpGeom.getInverseProjectionMatrix()
                                                     - geom->getInverseProjectionMatrix())
                                                        .norm();
                        REQUIRE_UNARY(checkApproxEq(centerNorm, 0));
                        REQUIRE_UNARY(checkApproxEq(projMatNorm, 0, 0.0000001));
                        REQUIRE_UNARY(checkApproxEq(invProjMatNorm, 0, 0.0000001));
                    }
                }
            }
        }
    }
}

TEST_CASE("LimitedAngleTrajectoryGenerator: Create a non-mirrored Limited Angle Trajectory")
{
    using namespace geometry;

    const index_t s = 64;

    // Detector size is the volume size scaled by the square root of 2
    const auto expectedDetectorSize = static_cast<index_t>(s * std::sqrt(2));

    GIVEN("A 2D descriptor and 256 angles")
    {
        index_t numberOfAngles = 256;
        IndexVector_t volSize(2);
        volSize << s, s;
        VolumeDescriptor desc{volSize};

        WHEN("We create a half circular limited angle trajectory for this scenario")
        {
            index_t halfCircular = 180;
            real_t diffCenterSource{s * 100};
            real_t diffCenterDetector{s};
            std::pair missingWedgeAngles(geometry::Degree(40), geometry::Degree(45));

            auto sinoDescriptor = LimitedAngleTrajectoryGenerator::createTrajectory(
                numberOfAngles, missingWedgeAngles, desc, halfCircular, diffCenterSource,
                diffCenterDetector, false);

            // check that the detector size is correct
            REQUIRE_EQ(sinoDescriptor->getNumberOfCoefficientsPerDimension()[0],
                       expectedDetectorSize);

            THEN("Every geomList in our list has the same camera center and the same projection "
                 "matrix")
            {
                const real_t sourceToCenter = diffCenterSource;
                const real_t centerToDetector = diffCenterDetector;

                real_t wedgeArc = missingWedgeAngles.second - missingWedgeAngles.first;

                const real_t angleIncrement = (static_cast<real_t>(halfCircular) - wedgeArc)
                                              / (static_cast<real_t>(numberOfAngles));

                index_t j = 0;
                for (index_t i = 0;; ++i) {
                    Radian angle = Degree{static_cast<real_t>(i) * angleIncrement};

                    if (angle.to_degree() > static_cast<real_t>(halfCircular)) {
                        break;
                    }

                    if (!(angle.to_degree() >= missingWedgeAngles.first
                          && angle.to_degree() <= missingWedgeAngles.second)) {
                        Geometry tmpGeom(SourceToCenterOfRotation{sourceToCenter},
                                         CenterOfRotationToDetector{centerToDetector},
                                         Radian{angle}, VolumeData2D{volSize},
                                         SinogramData2D{sinoDescriptor->getSpacingPerDimension(),
                                                        sinoDescriptor->getLocationOfOrigin()});

                        auto geom = sinoDescriptor->getGeometryAt(j++);
                        CHECK(geom);

                        const auto centerNorm =
                            (tmpGeom.getCameraCenter() - geom->getCameraCenter()).norm();
                        const auto projMatNorm =
                            (tmpGeom.getProjectionMatrix() - geom->getProjectionMatrix()).norm();
                        const auto invProjMatNorm = (tmpGeom.getInverseProjectionMatrix()
                                                     - geom->getInverseProjectionMatrix())
                                                        .norm();
                        REQUIRE_UNARY(checkApproxEq(centerNorm, 0));
                        REQUIRE_UNARY(checkApproxEq(projMatNorm, 0, 0.0000001));
                        REQUIRE_UNARY(checkApproxEq(invProjMatNorm, 0, 0.0000001));
                    }
                }
            }
        }

        WHEN("We create a full circular limited angle trajectory for this scenario")
        {
            index_t fullyCircular = 359;
            real_t diffCenterSource{s * 100};
            real_t diffCenterDetector{s};
            std::pair missingWedgeAngles(geometry::Degree(20), geometry::Degree(85));

            auto sinoDescriptor = LimitedAngleTrajectoryGenerator::createTrajectory(
                numberOfAngles, missingWedgeAngles, desc, fullyCircular, diffCenterSource,
                diffCenterDetector, false);

            // check that the detector size is correct
            REQUIRE_EQ(sinoDescriptor->getNumberOfCoefficientsPerDimension()[0],
                       expectedDetectorSize);

            THEN("Every geomList in our list has the same camera center and the same projection "
                 "matrix")
            {
                const real_t sourceToCenter = diffCenterSource;
                const real_t centerToDetector = diffCenterDetector;

                real_t wedgeArc = missingWedgeAngles.second - missingWedgeAngles.first;

                const real_t angleIncrement = (static_cast<real_t>(fullyCircular) - wedgeArc)
                                              / (static_cast<real_t>(numberOfAngles));

                index_t j = 0;
                for (index_t i = 0;; ++i) {
                    Radian angle = Degree{static_cast<real_t>(i) * angleIncrement};

                    if (angle.to_degree() > static_cast<real_t>(fullyCircular)) {
                        break;
                    }

                    if (!(angle.to_degree() >= missingWedgeAngles.first
                          && angle.to_degree() <= missingWedgeAngles.second)) {
                        Geometry tmpGeom(SourceToCenterOfRotation{sourceToCenter},
                                         CenterOfRotationToDetector{centerToDetector},
                                         Radian{angle}, VolumeData2D{volSize},
                                         SinogramData2D{sinoDescriptor->getSpacingPerDimension(),
                                                        sinoDescriptor->getLocationOfOrigin()});

                        auto geom = sinoDescriptor->getGeometryAt(j++);
                        CHECK(geom);

                        const auto centerNorm =
                            (tmpGeom.getCameraCenter() - geom->getCameraCenter()).norm();
                        const auto projMatNorm =
                            (tmpGeom.getProjectionMatrix() - geom->getProjectionMatrix()).norm();
                        const auto invProjMatNorm = (tmpGeom.getInverseProjectionMatrix()
                                                     - geom->getInverseProjectionMatrix())
                                                        .norm();
                        REQUIRE_UNARY(checkApproxEq(centerNorm, 0));
                        REQUIRE_UNARY(checkApproxEq(projMatNorm, 0, 0.0000001));
                        REQUIRE_UNARY(checkApproxEq(invProjMatNorm, 0, 0.0000001));
                    }
                }
            }
        }
    }
}
