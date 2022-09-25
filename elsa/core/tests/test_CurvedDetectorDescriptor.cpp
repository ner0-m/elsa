/**
 * @file test_CurvedDetectorDescriptor.cpp
 *
 * @brief Test for CurvedDetectorDescriptor
 *
 * @author David Frank - initial code for PlanarDetectorDescriptor
 * @author Julia Spindler, Robert Imschweiler - adapt for CurvedDetectorDescriptor
 */

#include "StrongTypes.h"
#include "doctest/doctest.h"

#include "CurvedDetectorDescriptor.h"
#include "VolumeDescriptor.h"
#include <iostream>

using namespace elsa;
using namespace elsa::geometry;
using namespace doctest;

TEST_SUITE_BEGIN("core");

const geometry::Radian angleFakePlanar{static_cast<real_t>(1e-10)};

TEST_CASE("CurvedDetectorDescriptor: Testing 2D CurvedDetectorDescriptor as fake "
          "PlanarDetectorDescriptor")
{
    GIVEN("Given a 5x5 Volume and a single 5 wide detector pose")
    {
        IndexVector_t volSize(2);
        volSize << 5, 5;
        VolumeDescriptor ddVol(volSize);

        IndexVector_t sinoSize(2);
        sinoSize << 5, 1;

        real_t s2c = 10;
        real_t c2d = 4;

        Geometry g(SourceToCenterOfRotation{s2c}, CenterOfRotationToDetector{c2d}, Radian{0},
                   VolumeData2D{Size2D{volSize}}, SinogramData2D{Size2D{sinoSize}});

        CurvedDetectorDescriptor desc(sinoSize, {g}, angleFakePlanar, s2c + c2d);

        WHEN("Retreiving the single geometry pose")
        {
            auto geom = desc.getGeometryAt(0);
            auto geomList = desc.getGeometry();

            CHECK_EQ(desc.getNumberOfGeometryPoses(), 1);
            CHECK_EQ(geomList.size(), 1);

            THEN("Geometry is equal")
            {
                CHECK_EQ((geom), g);
                CHECK_EQ(geomList[0], g);
            }
        }

        WHEN("Generating rays for detecor pixels 0, 2 and 4")
        {
            for (real_t detPixel : std::initializer_list<real_t>{0, 2, 4}) {
                RealVector_t pixel(1);
                pixel << detPixel + 0.5f;

                // Check that ray for IndexVector_t is equal to previous one
                auto ray = desc.computeRayFromDetectorCoord(pixel, 0);

                // Create variables, which make typing quicker
                auto ro = ray.origin();
                auto rd = ray.direction();

                // Check that ray origin is camera center
                auto c = g.getCameraCenter();
                CHECK_EQ((ro - c).sum(), Approx(0));

                // compute intersection manually
                real_t t = Approx(rd[0]) == 0 ? (s2c + c2d) : ((pixel[0] - ro[0]) / rd[0]);

                auto detCoordY = ro[1] + t * rd[1];

                CHECK_EQ(detCoordY, Approx(ddVol.getLocationOfOrigin()[1] + c2d));
            }
        }

        GIVEN("Given a 5x5 Volume and a multiple 5 wide detector pose")
        {
            IndexVector_t volSize(2);
            volSize << 5, 5;
            VolumeDescriptor ddVol(volSize);

            IndexVector_t sinoSize(2);
            sinoSize << 5, 4;

            Geometry g1(SourceToCenterOfRotation{s2c}, CenterOfRotationToDetector{c2d}, Degree{0},
                        VolumeData2D{Size2D{volSize}}, SinogramData2D{Size2D{sinoSize}});
            Geometry g2(SourceToCenterOfRotation{s2c}, CenterOfRotationToDetector{c2d}, Degree{90},
                        VolumeData2D{Size2D{volSize}}, SinogramData2D{Size2D{sinoSize}});
            Geometry g3(SourceToCenterOfRotation{s2c}, CenterOfRotationToDetector{c2d}, Degree{180},
                        VolumeData2D{Size2D{volSize}}, SinogramData2D{Size2D{sinoSize}});
            Geometry g4(SourceToCenterOfRotation{s2c}, CenterOfRotationToDetector{c2d}, Degree{270},
                        VolumeData2D{Size2D{volSize}}, SinogramData2D{Size2D{sinoSize}});

            CurvedDetectorDescriptor desc(sinoSize, {g1, g2, g3, g4}, angleFakePlanar, s2c + c2d);

            WHEN("Retreiving geometry poses")
            {
                auto geom = desc.getGeometryAt(0);
                auto geomList = desc.getGeometry();

                CHECK_EQ(desc.getNumberOfGeometryPoses(), 4);
                CHECK_EQ(geomList.size(), 4);

                THEN("Geometry is equal")
                {
                    CHECK_EQ((desc.getGeometryAt(0)), g1);
                    CHECK_EQ(geomList[0], g1);
                }
                THEN("Geometry is equal")
                {
                    CHECK_EQ((desc.getGeometryAt(1)), g2);
                    CHECK_EQ(geomList[1], g2);
                }
                THEN("Geometry is equal")
                {
                    CHECK_EQ((desc.getGeometryAt(2)), g3);
                    CHECK_EQ(geomList[2], g3);
                }
                THEN("Geometry is equal")
                {
                    CHECK_EQ((desc.getGeometryAt(3)), g4);
                    CHECK_EQ(geomList[3], g4);
                }
            }

            WHEN("Check for multiple poses, that all the overloads compute the same rays")
            {
                for (index_t pose : {0, 1, 2, 3}) {

                    for (index_t detPixel : {0, 2, 4}) {
                        IndexVector_t pixel(2);
                        pixel << detPixel, pose;

                        RealVector_t pixelReal(1);
                        pixelReal << static_cast<real_t>(detPixel) + 0.5f;

                        auto ray1 =
                            desc.computeRayFromDetectorCoord(desc.getIndexFromCoordinate(pixel));
                        auto ray2 = desc.computeRayFromDetectorCoord(pixel);
                        auto ray3 = desc.computeRayFromDetectorCoord(pixelReal, pose);

                        auto ro1 = ray1.origin();
                        auto rd1 = ray1.direction();

                        auto ro2 = ray2.origin();
                        auto rd2 = ray2.direction();

                        auto ro3 = ray3.origin();
                        auto rd3 = ray3.direction();

                        CHECK_EQ(ro1, ro2);
                        CHECK_EQ(ro1, ro3);

                        CHECK_EQ(rd1, rd2);
                        CHECK_EQ(rd1, rd3);

                        // Shouldn't be necessary, but whatever
                        CHECK_EQ(ro2, ro3);
                        CHECK_EQ(rd2, rd3);
                    }
                }
            }
        }
    }
}

TEST_CASE("CurvedDetectorDescriptor: Testing 3D CurvedDetectorDescriptor as fake "
          "PlanarDetectorDescriptor")
{
    GIVEN("Given a 5x5x5 Volume and a single 5x5 wide detector pose")
    {
        IndexVector_t volSize(3);
        volSize << 5, 5, 5;
        VolumeDescriptor ddVol(volSize);

        IndexVector_t sinoSize(3);
        sinoSize << 5, 5, 1;

        real_t s2c = 10;
        real_t c2d = 4;

        Geometry g(SourceToCenterOfRotation{s2c}, CenterOfRotationToDetector{c2d},
                   VolumeData3D{Size3D{volSize}}, SinogramData3D{Size3D{sinoSize}},
                   RotationAngles3D{Gamma{0}});

        CurvedDetectorDescriptor desc(sinoSize, {g}, angleFakePlanar, s2c + c2d);

        WHEN("Retreiving the single geometry pose")
        {
            auto geom = desc.getGeometryAt(0);
            auto geomList = desc.getGeometry();

            CHECK_EQ(desc.getNumberOfGeometryPoses(), 1);
            CHECK_EQ(geomList.size(), 1);

            THEN("Geometry is equal")
            {
                CHECK_EQ((geom), g);
                CHECK_EQ(geomList[0], g);
            }
        }

        WHEN("Generating rays for detecor pixels 0, 2 and 4 for each dim")
        {
            for (index_t detPixel1 : {0, 2, 4}) {
                for (index_t detPixel2 : {0, 2, 4}) {
                    RealVector_t pixel(2);
                    pixel << static_cast<real_t>(detPixel1) + 0.5f,
                        static_cast<real_t>(detPixel2) + 0.5f;

                    // Check that ray for IndexVector_t is equal to previous one
                    auto ray = desc.computeRayFromDetectorCoord(pixel, 0);

                    // Create variables, which make typing quicker
                    auto ro = ray.origin();
                    auto rd = ray.direction();

                    // Check that ray origin is camera center
                    auto c = g.getCameraCenter();
                    CHECK_EQ((ro - c).sum(), Approx(0));

                    auto o = ddVol.getLocationOfOrigin();
                    RealVector_t detCoordWorld(3);
                    detCoordWorld << pixel[0] - o[0], pixel[1] - o[1], c2d;
                    RealVector_t rotD = g.getRotationMatrix().transpose() * detCoordWorld + o;

                    real_t factor = 0;
                    if (std::abs(rd[0]) > 0)
                        factor = (rotD[0] - ro[0]) / rd[0];
                    else if (std::abs(rd[1]) > 0)
                        factor = (rotD[1] - ro[1]) / rd[1];
                    else if (std::abs(rd[2]) > 0)
                        factor = (rotD[2] - ro[2] / rd[2]);

                    CHECK_EQ((ro[0] + factor * rd[0]), Approx(rotD[0]));
                    CHECK_EQ((ro[1] + factor * rd[1]), Approx(rotD[1]));
                    CHECK_EQ((ro[2] + factor * rd[2]), Approx(rotD[2]));
                }
            }
        }
    }
}

TEST_CASE("CurvedDetectorDescriptor: Testing 2D CurvedDetectorDescriptor translation to planar "
          "coordinates")
{
    real_t angle{2.0};
    real_t s2d{14.0};

    GIVEN("Given a single 5 wide detector pose")
    {
        IndexVector_t sinoSize(2);
        sinoSize << 5, 1;

        CurvedDetectorDescriptor desc(sinoSize, {}, geometry::Radian(angle), s2d);

        const std::vector<RealVector_t>& planarCoords{desc.getPlanarCoords()};

        real_t radius{static_cast<real_t>(sinoSize[0]) / angle};
        CHECK_EQ(radius, desc.getRadius());

        /*
         * The detector has 5 pixels
         *  -> the pixel in the center has index 2
         *  -> its center has the x-coordinate 2.5
         */
        CHECK_EQ(planarCoords[2][0], 2.5);

        // Towards the edges the pixel size gets smaller
        CHECK_EQ(planarCoords[0][0], Approx(0.60392));
        CHECK_EQ(planarCoords[1][0], Approx(1.51253));
        CHECK_EQ(planarCoords[3][0], Approx(3.48747));
        CHECK_EQ(planarCoords[4][0], Approx(4.39608));

        // Check that the spacing is symmetrical
        CHECK_EQ(2.5 - planarCoords[0][0], Approx(abs(2.5 - planarCoords[4][0])));
        CHECK_EQ(2.5 - planarCoords[1][0], Approx(abs(2.5 - planarCoords[3][0])));
    }

    GIVEN("Given a single 6 wide detector pose")
    {
        IndexVector_t sinoSize(2);
        sinoSize << 6, 1;

        CurvedDetectorDescriptor desc(sinoSize, {}, geometry::Radian(angle), s2d);

        const std::vector<RealVector_t>& planarCoords{desc.getPlanarCoords()};

        real_t radius{static_cast<real_t>(sinoSize[0]) / angle};
        CHECK_EQ(radius, desc.getRadius());

        /*
         * The detector has 6 pixels
         *  -> there is no pixel in the center
         *  -> the center is at 3
         */

        // Towards the outer pixels, the pixel size gets smaller.
        CHECK_EQ(planarCoords[0][0], Approx(0.61183));
        CHECK_EQ(planarCoords[1][0], Approx(1.52298));
        CHECK_EQ(planarCoords[2][0], Approx(2.50083));
        CHECK_EQ(planarCoords[3][0], Approx(3.49917));
        CHECK_EQ(planarCoords[4][0], Approx(4.47702));
        CHECK_EQ(planarCoords[5][0], Approx(5.38817));

        // check that the spacing is symmetrical
        CHECK_EQ(3.0 - planarCoords[0][0], Approx(abs(3.0 - planarCoords[5][0])));
        CHECK_EQ(3.0 - planarCoords[1][0], Approx(abs(3.0 - planarCoords[4][0])));
        CHECK_EQ(3.0 - planarCoords[2][0], Approx(abs(3.0 - planarCoords[3][0])));
    }
}

TEST_CASE("CurvedDetectorDescriptor: Testing 3D CurvedDetectorDescriptor translation to planar "
          "coordinates")
{
    real_t angle{2.0};
    real_t s2d{14.0};

    GIVEN("Given a single 5x5 wide detector pose")
    {
        IndexVector_t sinoSize(3);
        sinoSize << 5, 5, 1;

        CurvedDetectorDescriptor desc(sinoSize, {}, geometry::Radian(angle), s2d);

        const std::vector<RealVector_t>& planarCoords{desc.getPlanarCoords()};

        real_t radius{static_cast<real_t>(sinoSize[0]) / angle};
        CHECK_EQ(radius, desc.getRadius());

        /*
         * The detector has 5x5 pixels
         * -> the column in the center starts at index 2
         * -> x_value stays the same, y_value counts upwards
         *  (y_value denotes the width of the detector)
         */
        float x_value = 2.5;
        float y_value_start = 0.5f;
        CHECK_EQ(planarCoords[2][0], Approx(x_value));
        CHECK_EQ(planarCoords[2][1], Approx(y_value_start));
        CHECK_EQ(planarCoords[7][0], Approx(x_value));
        CHECK_EQ(planarCoords[7][1], Approx(y_value_start + 1));
        CHECK_EQ(planarCoords[12][0], Approx(x_value));
        CHECK_EQ(planarCoords[12][1], Approx(y_value_start + 2));
        CHECK_EQ(planarCoords[17][0], Approx(x_value));
        CHECK_EQ(planarCoords[17][1], Approx(y_value_start + 3));
        CHECK_EQ(planarCoords[22][0], Approx(x_value));
        CHECK_EQ(planarCoords[22][1], Approx(y_value_start + 4));

        // check for symmetry in the last row and last column
        CHECK_EQ(2.5 - planarCoords[20][0], Approx(abs(2.5 - planarCoords[24][0])));
        CHECK_EQ(2.5 - planarCoords[21][0], Approx(abs(2.5 - planarCoords[23][0])));
        CHECK_EQ(2.5 - planarCoords[9][1], Approx(abs(2.5 - planarCoords[19][1])));
        CHECK_EQ(2.5 - planarCoords[4][1], Approx(abs(2.5 - planarCoords[24][1])));

        // check that the x-value deltas get smaller towards the edges as the source is far away
        CHECK_LT(planarCoords[1][0] - planarCoords[0][0], planarCoords[2][0] - planarCoords[1][0]);
        CHECK_LT(planarCoords[4][0] - planarCoords[3][0], planarCoords[2][0] - planarCoords[1][0]);

        // check that the y-value deltas get bigger towards the edges
        CHECK_LT(planarCoords[0][1], planarCoords[1][1]);
        CHECK_GT(planarCoords[20][1], planarCoords[21][1]);
    }

    GIVEN("Given a single 6x8 wide detector pose")
    {
        IndexVector_t sinoSize(3);
        sinoSize << 6, 8, 1;

        CurvedDetectorDescriptor desc(sinoSize, {}, geometry::Radian(angle), s2d);

        const std::vector<RealVector_t>& planarCoords{desc.getPlanarCoords()};

        real_t radius{static_cast<real_t>(sinoSize[0]) / angle};
        CHECK_EQ(radius, desc.getRadius());

        /*
         * The detector has 6 pixels
         *  -> there is no pixel in the center
         */

        // check for symmetry in the last row and last column
        CHECK_EQ(3 - planarCoords[42][0], Approx(abs(3 - planarCoords[47][0])));
        CHECK_EQ(3 - planarCoords[43][0], Approx(abs(3 - planarCoords[46][0])));
        CHECK_EQ(3 - planarCoords[44][0], Approx(abs(3 - planarCoords[45][0])));
        CHECK_EQ(4 - planarCoords[5][1], Approx(abs(4 - planarCoords[47][1])));
        CHECK_EQ(4 - planarCoords[11][1], Approx(abs(4 - planarCoords[41][1])));
        CHECK_EQ(4 - planarCoords[17][1], Approx(abs(4 - planarCoords[35][1])));
        CHECK_EQ(4 - planarCoords[23][1], Approx(abs(4 - planarCoords[29][1])));

        // check that the x-value deltas get smaller towards the edges
        CHECK_LT(planarCoords[1][0] - planarCoords[0][0], planarCoords[3][0] - planarCoords[2][0]);
        CHECK_LT(planarCoords[5][0] - planarCoords[4][0], planarCoords[3][0] - planarCoords[2][0]);

        // check that the y-value deltas get bigger towards the edges
        CHECK_LT(planarCoords[0][1], planarCoords[1][1]);
        CHECK_GT(planarCoords[42][1], planarCoords[43][1]);
    }
}

TEST_SUITE_END();
