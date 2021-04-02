/**
 * @file test_PlanarDetectorDescriptor.cpp
 *
 * @brief Test for PlanarDetectorDescriptor
 *
 * @author David Frank - initial code
 */

#include "doctest/doctest.h"

#include "PlanarDetectorDescriptor.h"
#include "VolumeDescriptor.h"
#include "Logger.h" 

using namespace elsa;
using namespace elsa::geometry;
using namespace doctest;

using Ray = DetectorDescriptor::Ray;

TEST_SUITE_BEGIN("core");

TEST_CASE("PlanarDetectorDescriptor: Testing 2D PlanarDetectorDescriptor")
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

        PlanarDetectorDescriptor desc(sinoSize, {g});

        WHEN("Retreiving the single geometry pose")
        {
            auto geom = desc.getGeometryAt(0);

            CHECK_EQ(desc.getNumberOfGeometryPoses(), 1);

            THEN("Geometry is equal") { CHECK_EQ((geom), g); }
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
        WHEN("Computing detector coord from ray")
        {
            auto ro = g.getCameraCenter();
            auto rd = RealVector_t(2);

            THEN("The detector coord is correct")
            {
                rd << -0.141421f, 0.989949f;
                rd.normalize();

                auto detCoord = desc.computeDetectorCoordFromRay(Ray(ro, rd), 0);
                CHECK_EQ(detCoord[0], Approx(0.5).epsilon(0.05));
            }
            THEN("The detector coord is correct asd")
            {
                rd << 0.f, 1.f;
                rd.normalize();

                auto detCoord = desc.computeDetectorCoordFromRay(Ray(ro, rd), 0);
                CHECK_EQ(detCoord[0], Approx(2.5));
            }

            THEN("The detector coord is correct asd")
            {
                rd << 0.141421f, 0.989949f;
                rd.normalize();

                auto detCoord = desc.computeDetectorCoordFromRay(Ray(ro, rd), 0);
                CHECK_EQ(detCoord[0], Approx(4.5).epsilon(0.05));
            }
        }

        GIVEN("Given a 5x5 Volume and a multiple 5 wide detector pose")
        {
            IndexVector_t volSize(2);
            volSize << 5, 5;
            VolumeDescriptor ddVol(volSize);

            IndexVector_t sinoSize(2);
            sinoSize << 5, 4;

            real_t s2c = 10;
            real_t c2d = 4;

            Geometry g1(SourceToCenterOfRotation{s2c}, CenterOfRotationToDetector{c2d}, Degree{0},
                        VolumeData2D{Size2D{volSize}}, SinogramData2D{Size2D{sinoSize}});
            Geometry g2(SourceToCenterOfRotation{s2c}, CenterOfRotationToDetector{c2d}, Degree{90},
                        VolumeData2D{Size2D{volSize}}, SinogramData2D{Size2D{sinoSize}});
            Geometry g3(SourceToCenterOfRotation{s2c}, CenterOfRotationToDetector{c2d}, Degree{180},
                        VolumeData2D{Size2D{volSize}}, SinogramData2D{Size2D{sinoSize}});
            Geometry g4(SourceToCenterOfRotation{s2c}, CenterOfRotationToDetector{c2d}, Degree{270},
                        VolumeData2D{Size2D{volSize}}, SinogramData2D{Size2D{sinoSize}});

            PlanarDetectorDescriptor desc(sinoSize, {g1, g2, g3, g4});

            WHEN("Retreiving the single geometry pose")
            {
                auto geom = desc.getGeometryAt(0);

                CHECK_EQ(desc.getNumberOfGeometryPoses(), 4);

                THEN("Geometry is equal") { CHECK_EQ((desc.getGeometryAt(0)), g1); }
                THEN("Geometry is equal") { CHECK_EQ((desc.getGeometryAt(1)), g2); }
                THEN("Geometry is equal") { CHECK_EQ((desc.getGeometryAt(2)), g3); }
                THEN("Geometry is equal") { CHECK_EQ((desc.getGeometryAt(3)), g4); }
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

TEST_CASE("PlanarDetectorDescriptor: Testing 3D PlanarDetectorDescriptor")
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

        PlanarDetectorDescriptor desc(sinoSize, {g});

        WHEN("Retreiving the single geometry pose")
        {
            auto geom = desc.getGeometryAt(0);

            CHECK_EQ(desc.getNumberOfGeometryPoses(), 1);

            THEN("Geometry is equal") { CHECK_EQ((geom), g); }
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

TEST_CASE("PlanarDetectorDescriptor: Printing PlanarDetectorDescriptor")
{
    // FIXME: Actually test the formatting, this only tests compilation
    IndexVector_t volSize(3);
    volSize << 5, 5, 5;

    IndexVector_t sinoSize(3);
    sinoSize << 5, 5, 1;

    const real_t s2c = 10;
    const real_t c2d = 4;

    const Geometry g(SourceToCenterOfRotation{s2c}, CenterOfRotationToDetector{c2d},
                     VolumeData3D{Size3D{volSize}}, SinogramData3D{Size3D{sinoSize}},
                     RotationAngles3D{Gamma{0}});

    PlanarDetectorDescriptor desc(sinoSize, {g});

    infoln("Printing without testing: {}", desc);
}

TEST_SUITE_END();
