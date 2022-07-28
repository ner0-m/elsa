#include "doctest/doctest.h"

#include "JosephsMethod.h"
#include "Geometry.h"
#include "Logger.h"
#include "testHelpers.h"
#include "VolumeDescriptor.h"
#include "PlanarDetectorDescriptor.h"

using namespace elsa;
using namespace elsa::geometry;
using namespace doctest;

// TODO(dfrank): remove this and replace with checkApproxEq
using doctest::Approx;

TEST_CASE("JosephsMethod: Testing with only one ray")
{
    // Turn logger of
    Logger::setLevel(Logger::LogLevel::OFF);

    IndexVector_t sizeDomain(2);
    sizeDomain << 5, 5;

    IndexVector_t sizeRange(2);
    sizeRange << 1, 1;

    auto domain = VolumeDescriptor(sizeDomain);

    auto stc = SourceToCenterOfRotation{100};
    auto ctr = CenterOfRotationToDetector{5};
    auto volData = VolumeData2D{Size2D{sizeDomain}};
    auto sinoData = SinogramData2D{Size2D{sizeRange}};

    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "",
                                 " << ", ";");

    GIVEN("A JosephsMethod for 2D and a domain data with all ones")
    {
        std::vector<Geometry> geom;

        auto dataDomain = DataContainer(domain);
        dataDomain = 1;

        WHEN("We have a single ray with 0 degrees")
        {
            geom.emplace_back(stc, ctr, Radian{0}, std::move(volData), std::move(sinoData));

            auto detectorDesc = PlanarDetectorDescriptor(sizeRange, {geom});

            auto op = JosephsMethod(domain, detectorDesc);

            THEN("A^t A x should be close to the original data")
            {
                auto AtAx = DataContainer(domain);
                auto dataRange = DataContainer(detectorDesc);
                dataRange = 0;

                op.apply(dataDomain, dataRange);

                REQUIRE_EQ(dataRange[0], Approx(5));

                op.applyAdjoint(dataRange, AtAx);

                auto cmp = RealVector_t(sizeDomain.prod());
                cmp << 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0;

                REQUIRE_UNARY(isCwiseApprox(DataContainer(domain, cmp), AtAx));
            }
        }

        WHEN("We have a single ray with 180 degrees")
        {
            geom.emplace_back(stc, ctr, Degree{180}, std::move(volData), std::move(sinoData));

            auto detectorDesc = PlanarDetectorDescriptor(sizeRange, {geom});

            auto op = JosephsMethod(domain, detectorDesc);

            THEN("A^t A x should be close to the original data")
            {
                auto AtAx = DataContainer(domain);
                auto dataRange = DataContainer(detectorDesc);
                dataRange = 0;

                op.apply(dataDomain, dataRange);

                REQUIRE_EQ(dataRange[0], Approx(5));

                op.applyAdjoint(dataRange, AtAx);

                auto cmp = RealVector_t(sizeDomain.prod());
                cmp << 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0;

                REQUIRE_UNARY(isApprox(DataContainer(domain, cmp), AtAx));
            }
        }

        WHEN("We have a single ray with 90 degrees")
        {
            geom.emplace_back(stc, ctr, Degree{90}, std::move(volData), std::move(sinoData));

            auto detectorDesc = PlanarDetectorDescriptor(sizeRange, {geom});

            auto op = JosephsMethod(domain, detectorDesc);

            THEN("A^t A x should be close to the original data")
            {
                auto AtAx = DataContainer(domain);
                auto dataRange = DataContainer(detectorDesc);
                dataRange = 0;

                op.apply(dataDomain, dataRange);

                REQUIRE_EQ(dataRange[0], Approx(5));

                op.applyAdjoint(dataRange, AtAx);

                auto cmp = RealVector_t(sizeDomain.prod());
                cmp << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

                REQUIRE_UNARY(isApprox(DataContainer(domain, cmp), AtAx));
            }
        }

        WHEN("We have a single ray with 270 degrees")
        {
            geom.emplace_back(stc, ctr, Degree{270}, std::move(volData), std::move(sinoData));

            auto detectorDesc = PlanarDetectorDescriptor(sizeRange, {geom});

            auto op = JosephsMethod(domain, detectorDesc);

            THEN("A^t A x should be close to the original data")
            {
                auto AtAx = DataContainer(domain);
                auto dataRange = DataContainer(detectorDesc);
                dataRange = 0;

                op.apply(dataDomain, dataRange);

                REQUIRE_EQ(dataRange[0], Approx(5));

                op.applyAdjoint(dataRange, AtAx);

                auto cmp = RealVector_t(sizeDomain.prod());
                cmp << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

                REQUIRE_UNARY(isApprox(DataContainer(domain, cmp), AtAx));
            }
        }

        WHEN("We have a single ray with 45 degrees")
        {
            geom.emplace_back(stc, ctr, Degree{45}, std::move(volData), std::move(sinoData));

            auto detectorDesc = PlanarDetectorDescriptor(sizeRange, {geom});

            auto op = JosephsMethod(domain, detectorDesc);

            THEN("A^t A x should be close to the original data")
            {
                auto AtAx = DataContainer(domain);
                auto dataRange = DataContainer(detectorDesc);
                dataRange = 0;

                op.apply(dataDomain, dataRange);
                op.applyAdjoint(dataRange, AtAx);

                auto cmp = RealVector_t(sizeDomain.prod());
                cmp << 10, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0,
                    10;

                REQUIRE_UNARY(isApprox(DataContainer(domain, cmp), AtAx, epsilon));
            }
        }

        WHEN("We have a single ray with 225 degrees")
        {
            geom.emplace_back(stc, ctr, Degree{225}, std::move(volData), std::move(sinoData));

            auto detectorDesc = PlanarDetectorDescriptor(sizeRange, {geom});

            auto op = JosephsMethod(domain, detectorDesc);

            THEN("A^t A x should be close to the original data")
            {
                auto AtAx = DataContainer(domain);
                auto dataRange = DataContainer(detectorDesc);
                dataRange = 0;

                op.apply(dataDomain, dataRange);
                op.applyAdjoint(dataRange, AtAx);

                auto cmp = RealVector_t(sizeDomain.prod());
                cmp << 10, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0,
                    10;

                REQUIRE_UNARY(isApprox(DataContainer(domain, cmp), AtAx, epsilon));
            }
        }
    }
}

TEST_CASE("JosephsMethod: Testing with only 4 ray")
{
    // Turn logger of
    Logger::setLevel(Logger::LogLevel::OFF);

    IndexVector_t sizeDomain(2);
    sizeDomain << 5, 5;

    IndexVector_t sizeRange(2);
    sizeRange << 1, 4;

    auto domain = VolumeDescriptor(sizeDomain);

    auto stc = SourceToCenterOfRotation{100};
    auto ctr = CenterOfRotationToDetector{5};
    auto volData = VolumeData2D{Size2D{sizeDomain}};
    auto sinoData = SinogramData2D{Size2D{sizeRange}};

    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "",
                                 " << ", ";");

    GIVEN("A JosephsMethod for 2D and a domain data with all ones")
    {
        std::vector<Geometry> geom;

        auto dataDomain = DataContainer(domain);
        dataDomain = 1;

        WHEN("We have a single ray with 0, 90, 180, 270 degrees")
        {
            geom.emplace_back(stc, ctr, Degree{0}, VolumeData2D{Size2D{sizeDomain}},
                              SinogramData2D{Size2D{sizeRange}});
            geom.emplace_back(stc, ctr, Degree{90}, VolumeData2D{Size2D{sizeDomain}},
                              SinogramData2D{Size2D{sizeRange}});
            geom.emplace_back(stc, ctr, Degree{180}, VolumeData2D{Size2D{sizeDomain}},
                              SinogramData2D{Size2D{sizeRange}});
            geom.emplace_back(stc, ctr, Degree{270}, VolumeData2D{Size2D{sizeDomain}},
                              SinogramData2D{Size2D{sizeRange}});

            auto detectorDesc = PlanarDetectorDescriptor(sizeRange, geom);
            auto op = JosephsMethod(domain, detectorDesc);

            auto dataRange = DataContainer(detectorDesc);
            dataRange = 0;

            THEN("A^t A x should be close to the original data")
            {
                auto AtAx = DataContainer(domain);

                op.apply(dataDomain, dataRange);
                op.applyAdjoint(dataRange, AtAx);

                auto cmp = RealVector_t(sizeDomain.prod());
                cmp << 0, 0, 10, 0, 0, 0, 0, 10, 0, 0, 10, 10, 20, 10, 10, 0, 0, 10, 0, 0, 0, 0, 10,
                    0, 0;

                REQUIRE_UNARY(isApprox(DataContainer(domain, cmp), AtAx));
            }
        }
    }
}

TEST_CASE("JosephsMethod: Calls to functions of super class")
{
    // Turn logger of
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("A projector")
    {
        IndexVector_t volumeDims(2), sinoDims(2);
        const index_t volSize = 10;
        const index_t detectorSize = 10;
        const index_t numImgs = 10;
        volumeDims << volSize, volSize;
        sinoDims << detectorSize, numImgs;
        VolumeDescriptor volumeDescriptor(volumeDims);

        DataContainer volume(volumeDescriptor);
        volume = 0;

        auto stc = SourceToCenterOfRotation{20 * volSize};
        auto ctr = CenterOfRotationToDetector{volSize};

        std::vector<Geometry> geom;
        for (std::size_t i = 0; i < numImgs; i++) {
            real_t angle = static_cast<real_t>(i * 2) * pi_t / 10;
            geom.emplace_back(stc, ctr, Radian{angle}, VolumeData2D{Size2D{volumeDims}},
                              SinogramData2D{Size2D{sinoDims}});
        }

        PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
        DataContainer sino(sinoDescriptor);
        sino = 0;

        JosephsMethod op(volumeDescriptor, sinoDescriptor);

        WHEN("Projector is cloned")
        {
            auto opClone = op.clone();
            auto sinoClone = sino;
            auto volumeClone = volume;

            THEN("Results do not change (may still be slightly different due to summation being "
                 "performed in a different order)")
            {
                op.apply(volume, sino);
                opClone->apply(volume, sinoClone);
                REQUIRE_UNARY(isCwiseApprox(sino, sinoClone));

                op.applyAdjoint(sino, volume);
                opClone->applyAdjoint(sino, volumeClone);
                REQUIRE_UNARY(isCwiseApprox(volume, volumeClone));
            }
        }
    }
}

TEST_CASE("JosephsMethod: Output DataContainer is not zero initialized")
{
    // Turn logger of
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("A 2D setting")
    {
        IndexVector_t volumeDims(2), sinoDims(2);
        const index_t volSize = 5;
        const index_t detectorSize = 1;
        const index_t numImgs = 1;
        volumeDims << volSize, volSize;
        sinoDims << detectorSize, numImgs;
        VolumeDescriptor volumeDescriptor(volumeDims);
        DataContainer volume(volumeDescriptor);

        auto stc = SourceToCenterOfRotation{20 * volSize};
        auto ctr = CenterOfRotationToDetector{volSize};
        auto volData = VolumeData2D{Size2D{volumeDims}};
        auto sinoData = SinogramData2D{Size2D{sinoDims}};

        std::vector<Geometry> geom;
        geom.emplace_back(stc, ctr, Radian{0}, std::move(volData), std::move(sinoData));

        PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
        DataContainer sino(sinoDescriptor);

        JosephsMethod op(volumeDescriptor, sinoDescriptor, JosephsMethod<>::Interpolation::LINEAR);

        WHEN("Sinogram container is not zero initialized and we project through an empty volume")
        {
            volume = 0;
            sino = 1;

            THEN("Result is zero")
            {
                op.apply(volume, sino);

                DataContainer zero(sinoDescriptor);
                zero = 0;
                REQUIRE_UNARY(isApprox(sino, zero, epsilon));
            }
        }

        WHEN("Volume container is not zero initialized and we backproject from an empty sinogram")
        {
            sino = 0;
            volume = 1;

            THEN("Result is zero")
            {
                op.applyAdjoint(sino, volume);

                DataContainer zero(volumeDescriptor);
                zero = 0;
                REQUIRE_UNARY(isApprox(volume, zero, epsilon));
            }
        }
    }

    GIVEN("A 3D setting")
    {
        IndexVector_t volumeDims(3), sinoDims(3);
        const index_t volSize = 3;
        const index_t detectorSize = 1;
        const index_t numImgs = 1;
        volumeDims << volSize, volSize, volSize;
        sinoDims << detectorSize, detectorSize, numImgs;
        VolumeDescriptor volumeDescriptor(volumeDims);
        DataContainer volume(volumeDescriptor);

        auto stc = SourceToCenterOfRotation{20 * volSize};
        auto ctr = CenterOfRotationToDetector{volSize};
        auto volData = VolumeData3D{Size3D{volumeDims}};
        auto sinoData = SinogramData3D{Size3D{sinoDims}};

        std::vector<Geometry> geom;

        geom.emplace_back(stc, ctr, std::move(volData), std::move(sinoData),
                          RotationAngles3D{Gamma{0}});
        PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);

        JosephsMethod op(volumeDescriptor, sinoDescriptor, JosephsMethod<>::Interpolation::LINEAR);

        DataContainer sino(sinoDescriptor);

        WHEN("Sinogram container is not zero initialized and we project through an empty volume")
        {
            volume = 0;
            sino = 1;

            THEN("Result is zero")
            {
                op.apply(volume, sino);
                DataContainer zero(sinoDescriptor);
                zero = 0;
                REQUIRE_UNARY(isApprox(sino, zero, epsilon));
            }
        }

        WHEN("Volume container is not zero initialized and we backproject from an empty sinogram")
        {
            sino = 0;
            volume = 1;

            THEN("Result is zero")
            {
                op.applyAdjoint(sino, volume);
                DataContainer zero(volumeDescriptor);
                zero = 0;
                REQUIRE_UNARY(isApprox(volume, zero, epsilon));
            }
        }
    }
}

TEST_CASE("JosephsMethod: Rays not intersecting the bounding box are present")
{
    // Turn logger of
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("A 2D setting")
    {
        IndexVector_t volumeDims(2), sinoDims(2);
        const index_t volSize = 5;
        const index_t detectorSize = 1;
        const index_t numImgs = 1;
        volumeDims << volSize, volSize;
        sinoDims << detectorSize, numImgs;
        VolumeDescriptor volumeDescriptor(volumeDims);
        DataContainer volume(volumeDescriptor);
        volume = 1;

        auto stc = SourceToCenterOfRotation{20 * volSize};
        auto ctr = CenterOfRotationToDetector{volSize};
        auto volData = VolumeData2D{Size2D{volumeDims}};
        auto sinoData = SinogramData2D{Size2D{sinoDims}};

        std::vector<Geometry> geom;

        WHEN("Tracing along a y-axis-aligned ray with a negative x-coordinate of origin")
        {
            geom.emplace_back(stc, ctr, Radian{0}, std::move(volData), std::move(sinoData),
                              PrincipalPointOffset{}, RotationOffset2D{volSize, 0});

            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);
            sino = 1;

            JosephsMethod op(volumeDescriptor, sinoDescriptor,
                             JosephsMethod<>::Interpolation::LINEAR);

            THEN("Result of forward projection is zero")
            {
                op.apply(volume, sino);
                DataContainer zero(sinoDescriptor);
                zero = 0;
                REQUIRE_UNARY(isApprox(sino, zero, epsilon));

                AND_THEN("Result of backprojection is zero")
                {
                    op.applyAdjoint(sino, volume);
                    DataContainer zero(volumeDescriptor);
                    zero = 0;
                    REQUIRE_UNARY(isApprox(volume, zero, epsilon));
                }
            }
        }

        WHEN("Tracing along a y-axis-aligned ray with a x-coordinate of origin beyond the bounding "
             "box")
        {
            geom.emplace_back(stc, ctr, Radian{0}, std::move(volData), std::move(sinoData),
                              PrincipalPointOffset{}, RotationOffset2D{-volSize, 0});

            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);
            sino = 1;

            JosephsMethod op(volumeDescriptor, sinoDescriptor,
                             JosephsMethod<>::Interpolation::LINEAR);

            THEN("Result of forward projection is zero")
            {
                op.apply(volume, sino);
                DataContainer zero(sinoDescriptor);
                zero = 0;
                REQUIRE_UNARY(isApprox(sino, zero, epsilon));

                AND_THEN("Result of backprojection is zero")
                {
                    op.applyAdjoint(sino, volume);
                    DataContainer zero(volumeDescriptor);
                    zero = 0;
                    REQUIRE_UNARY(isApprox(volume, zero, epsilon));
                }
            }
        }

        WHEN("Tracing along a x-axis-aligned ray with a negative y-coordinate of origin")
        {
            geom.emplace_back(stc, ctr, Radian{pi_t / 2}, std::move(volData), std::move(sinoData),
                              PrincipalPointOffset{}, RotationOffset2D{0, volSize});

            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);
            sino = 1;

            JosephsMethod op(volumeDescriptor, sinoDescriptor,
                             JosephsMethod<>::Interpolation::LINEAR);

            THEN("Result of forward projection is zero")
            {
                op.apply(volume, sino);
                DataContainer zero(sinoDescriptor);
                zero = 0;
                REQUIRE_UNARY(isApprox(sino, zero, epsilon));

                AND_THEN("Result of backprojection is zero")
                {
                    op.applyAdjoint(sino, volume);
                    DataContainer zero(volumeDescriptor);
                    zero = 0;
                    REQUIRE_UNARY(isApprox(volume, zero, epsilon));
                }
            }
        }

        WHEN("Tracing along a x-axis-aligned ray with a y-coordinate of origin beyond the bounding "
             "box")
        {
            geom.emplace_back(stc, ctr, Radian{pi_t / 2}, std::move(volData), std::move(sinoData),
                              PrincipalPointOffset{}, RotationOffset2D{0, -volSize});

            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);
            sino = 1;

            JosephsMethod op(volumeDescriptor, sinoDescriptor,
                             JosephsMethod<>::Interpolation::LINEAR);

            THEN("Result of forward projection is zero")
            {
                op.apply(volume, sino);
                DataContainer zero(sinoDescriptor);
                zero = 0;
                REQUIRE_UNARY(isApprox(sino, zero, epsilon));

                AND_THEN("Result of backprojection is zero")
                {
                    op.applyAdjoint(sino, volume);
                    DataContainer zero(volumeDescriptor);
                    zero = 0;
                    REQUIRE_UNARY(isApprox(volume, zero, epsilon));
                }
            }
        }
    }

    GIVEN("A 3D setting")
    {
        IndexVector_t volumeDims(3), sinoDims(3);
        const index_t volSize = 5;
        const index_t detectorSize = 1;
        const index_t numImgs = 1;
        volumeDims << volSize, volSize, volSize;
        sinoDims << detectorSize, detectorSize, numImgs;
        VolumeDescriptor volumeDescriptor(volumeDims);
        DataContainer volume(volumeDescriptor);
        volume = 1;

        auto stc = SourceToCenterOfRotation{20 * volSize};
        auto ctr = CenterOfRotationToDetector{volSize};

        std::vector<Geometry> geom;

        const index_t numCases = 9;
        real_t alpha[numCases] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        real_t beta[numCases] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, pi_t / 2, pi_t / 2, pi_t / 2};
        real_t gamma[numCases] = {0.0,      0.0,      0.0,      pi_t / 2, pi_t / 2,
                                  pi_t / 2, pi_t / 2, pi_t / 2, pi_t / 2};
        real_t offsetx[numCases] = {volSize, 0.0, volSize, 0.0, 0.0, 0.0, volSize, 0.0, volSize};
        real_t offsety[numCases] = {0.0, volSize, volSize, volSize, 0.0, volSize, 0.0, 0.0, 0.0};
        real_t offsetz[numCases] = {0.0, 0.0, 0.0, 0.0, volSize, volSize, 0.0, volSize, volSize};
        std::string neg[numCases] = {"x", "y", "x and y", "y", "z", "y and z", "x", "z", "x and z"};
        std::string ali[numCases] = {"z", "z", "z", "x", "x", "x", "y", "y", "y"};

        for (int i = 0; i < numCases; i++) {
            WHEN("Tracing along an axis-aligned ray with negative direction in one direction")
            {
                INFO("Tracing along a ", ali[i], "-axis-aligned ray with negative ",
                     neg[i] + "-coodinate of origin");
                geom.emplace_back(stc, ctr, VolumeData3D{Size3D{volumeDims}},
                                  SinogramData3D{Size3D{sinoDims}},
                                  RotationAngles3D{Gamma{gamma[i]}, Beta{beta[i]}, Alpha{alpha[i]}},
                                  PrincipalPointOffset2D{0, 0},
                                  RotationOffset3D{-offsetx[i], -offsety[i], -offsetz[i]});

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer sino(sinoDescriptor);
                sino = 1;

                JosephsMethod op(volumeDescriptor, sinoDescriptor,
                                 JosephsMethod<>::Interpolation::LINEAR);

                THEN("Result of forward projection is zero")
                {
                    op.apply(volume, sino);
                    DataContainer zero(sinoDescriptor);
                    zero = 0;
                    REQUIRE_UNARY(isApprox(sino, zero, epsilon));

                    AND_THEN("Result of backprojection is zero")
                    {
                        op.applyAdjoint(sino, volume);
                        DataContainer zero(volumeDescriptor);
                        zero = 0;
                        REQUIRE_UNARY(isApprox(volume, zero, epsilon));
                    }
                }
            }
        }
    }
}

TEST_CASE("JosephsMethod: Axis-aligned rays are present")
{
    // Turn logger of
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("A 2D setting with a single ray")
    {
        IndexVector_t volumeDims(2), sinoDims(2);
        const index_t volSize = 5;
        const index_t detectorSize = 1;
        const index_t numImgs = 1;
        volumeDims << volSize, volSize;
        sinoDims << detectorSize, numImgs;
        VolumeDescriptor volumeDescriptor(volumeDims);
        DataContainer volume(volumeDescriptor);

        auto stc = SourceToCenterOfRotation{20 * volSize};
        auto ctr = CenterOfRotationToDetector{volSize};
        auto volData = VolumeData2D{Size2D{volumeDims}};
        auto sinoData = SinogramData2D{Size2D{sinoDims}};

        std::vector<Geometry> geom;

        const index_t numCases = 4;
        const real_t angles[numCases] = {0.0, pi_t / 2, pi_t, 3 * pi_t / 2};
        RealVector_t backProj[2];
        backProj[0].resize(volSize * volSize);
        backProj[1].resize(volSize * volSize);
        backProj[1] << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

        backProj[0] << 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0;

        for (index_t i = 0; i < numCases; i++) {
            WHEN("An axis-aligned ray with fixed angles pass through the center of a pixel")
            {
                INFO("An axis-aligned ray with an angle of ", angles[i],
                     " radians passes through the center of a pixel");

                geom.emplace_back(stc, ctr, Radian{angles[i]}, std::move(volData),
                                  std::move(sinoData));

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer sino(sinoDescriptor);

                JosephsMethod op(volumeDescriptor, sinoDescriptor,
                                 JosephsMethod<>::Interpolation::LINEAR);
                THEN("The result of projecting through a pixel is exactly the pixel value")
                {
                    for (index_t j = 0; j < volSize; j++) {
                        volume = 0;
                        if (i % 2 == 0)
                            volume(volSize / 2, j) = 1;
                        else
                            volume(j, volSize / 2) = 1;

                        op.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx(1));
                    }

                    AND_THEN("The backprojection sets the values of all hit pixels to the detector "
                             "value")
                    {
                        op.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(
                            isApprox(volume, DataContainer(volumeDescriptor, backProj[i % 2])));
                    }
                }
            }
        }

        real_t offsetx[numCases] = {0.25, 0.0, 0.25, 0.0};
        real_t offsety[numCases] = {0.0, 0.25, 0.0, 0.25};

        backProj[0] << 0, 0.25, 0.75, 0, 0, 0, 0.25, 0.75, 0, 0, 0, 0.25, 0.75, 0, 0, 0, 0.25, 0.75,
            0, 0, 0, 0.25, 0.75, 0, 0;

        backProj[1] << 0, 0, 0, 0, 0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75, 0.75, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0;

        for (index_t i = 0; i < numCases; i++) {
            WHEN("An axis-aligned ray with fixed angle, which does not pass through the center of "
                 "a pixel")
            {
                INFO("An axis-aligned ray with an angle of ", angles[i],
                     " radians does not pass through the center of a pixel");

                geom.emplace_back(SourceToCenterOfRotation{volSize * 2000}, ctr, Radian{angles[i]},
                                  std::move(volData), std::move(sinoData), PrincipalPointOffset{0},
                                  RotationOffset2D{-offsetx[i], -offsety[i]});

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer sino(sinoDescriptor);

                JosephsMethod op(volumeDescriptor, sinoDescriptor,
                                 JosephsMethod<>::Interpolation::LINEAR);
                THEN("The result of projecting through a pixel is the interpolated value between "
                     "the two pixels closest to the ray")
                {
                    for (index_t j = 0; j < volSize; j++) {
                        volume = 0;
                        if (i % 2 == 0)
                            volume(volSize / 2, j) = 1;
                        else
                            volume(j, volSize / 2) = 1;

                        op.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx(0.75));
                    }

                    AND_THEN("The backprojection yields the exact adjoint")
                    {
                        sino[0] = 1;
                        op.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(
                            isApprox(volume, DataContainer(volumeDescriptor, backProj[i % 2])));
                    }
                }
            }
        }

        backProj[0] << 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;

        WHEN("A y-axis-aligned ray runs along the right volume boundary")
        {
            geom.emplace_back(SourceToCenterOfRotation{volSize * 2000}, ctr, Radian{0},
                              std::move(volData), std::move(sinoData), PrincipalPointOffset{0},
                              RotationOffset2D{volSize * 0.5, 0});

            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);

            JosephsMethod op(volumeDescriptor, sinoDescriptor);

            THEN("The result of projecting through a pixel is exactly the pixel's value (we mirror "
                 "values at the border for the purpose of interpolation)")
            {
                for (index_t j = 0; j < volSize; j++) {
                    volume = 0;
                    volume(volSize - 1, j) = 1;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0], Approx(1));
                }

                AND_THEN(
                    "The slow backprojection yields the exact adjoint, the fast backprojection "
                    "also yields the exact adjoint for a very distant x-ray source")
                {
                    sino[0] = 1;
                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isApprox(volume, DataContainer(volumeDescriptor, backProj[0])));
                }
            }
        }

        backProj[0] << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0;

        WHEN("A y-axis-aligned ray runs along the left volume boundary")
        {
            geom.emplace_back(SourceToCenterOfRotation{volSize * 2000}, ctr, Radian{0},
                              std::move(volData), std::move(sinoData), PrincipalPointOffset{0},
                              RotationOffset2D{-volSize / 2.0, 0});
            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);

            JosephsMethod op(volumeDescriptor, sinoDescriptor);

            THEN("The result of projecting through a pixel is exactly the pixel's value (we mirror "
                 "values at the border for the purpose of interpolation)")
            {
                for (index_t j = 0; j < volSize; j++) {
                    volume = 0;
                    volume(0, j) = 1;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0], Approx(1));
                }

                AND_THEN(
                    "The slow backprojection yields the exact adjoint, the fast backprojection "
                    "also yields the exact adjoint for a very distant x-ray source")
                {
                    sino[0] = 1;
                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isApprox(volume, DataContainer(volumeDescriptor, backProj[0])));
                }
            }
        }
    }

    GIVEN("A non-quadratic 2D volume")
    {
        IndexVector_t volumeDims(2), sinoDims(2);
        volumeDims << 5, 2;
        const index_t detectorSize = 1;
        const index_t numImgs = 1;
        sinoDims << detectorSize, numImgs;
        VolumeDescriptor volumeDescriptor(volumeDims);
        DataContainer volume(volumeDescriptor);

        auto stc = SourceToCenterOfRotation{20 * 5};
        auto ctr = CenterOfRotationToDetector{5};
        auto volData = VolumeData2D{Size2D{volumeDims}};
        auto sinoData = SinogramData2D{Size2D{sinoDims}};

        std::vector<Geometry> geom;

        WHEN("An axis-aligned ray enters (and leaves) through the shorter volume dimension")
        {
            geom.emplace_back(stc, ctr, Radian{0}, std::move(volData), std::move(sinoData),
                              PrincipalPointOffset{0},
                              RotationOffset2D{static_cast<real_t>(1.6), 0});

            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);

            JosephsMethod op(volumeDescriptor, sinoDescriptor);

            THEN("The results of forward projecting are correct")
            {
                volume = 0;
                volume(4, 0) = 1;
                volume(4, 1) = 1;

                op.apply(volume, sino);
                REQUIRE_EQ(sino[0], Approx(1.2));

                AND_THEN("The backprojection yields the correct result")
                {
                    sino[0] = 1;

                    volume = 0;
                    DataContainer bpExpected = volume;
                    bpExpected(4, 0) = static_cast<real_t>(0.6);
                    bpExpected(4, 1) = static_cast<real_t>(0.6);
                    bpExpected(3, 0) = static_cast<real_t>(0.4);
                    bpExpected(3, 1) = static_cast<real_t>(0.4);

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isApprox(volume, bpExpected));
                }
            }
        }
    }

    GIVEN("A 3D setting with a single ray")
    {
        IndexVector_t volumeDims(3), sinoDims(3);
        const index_t volSize = 3;
        const index_t detectorSize = 1;
        const index_t numImgs = 1;
        volumeDims << volSize, volSize, volSize;
        sinoDims << detectorSize, detectorSize, numImgs;
        VolumeDescriptor volumeDescriptor(volumeDims);
        DataContainer volume(volumeDescriptor);

        auto stc = SourceToCenterOfRotation{20 * volSize};
        auto ctr = CenterOfRotationToDetector{volSize};
        auto volData = VolumeData3D{Size3D{volumeDims}};
        auto sinoData = SinogramData3D{Size3D{sinoDims}};

        std::vector<Geometry> geom;

        const index_t numCases = 6;
        real_t beta[numCases] = {0.0, 0.0, 0.0, 0.0, pi_t / 2, 3 * pi_t / 2};
        real_t gamma[numCases] = {0.0, pi_t, pi_t / 2, 3 * pi_t / 2, pi_t / 2, 3 * pi_t / 2};
        std::string al[numCases] = {"z", "-z", "x", "-x", "y", "-y"};

        RealVector_t backProj[numCases];
        for (auto& backPr : backProj)
            backPr.resize(volSize * volSize * volSize);
        backProj[2] << 0, 0, 0, 0, 0, 0, 0, 0, 0,

            0, 1, 0, 0, 1, 0, 0, 1, 0,

            0, 0, 0, 0, 0, 0, 0, 0, 0;

        backProj[1] << 0, 0, 0, 0, 0, 0, 0, 0, 0,

            0, 0, 0, 1, 1, 1, 0, 0, 0,

            0, 0, 0, 0, 0, 0, 0, 0, 0;

        backProj[0] << 0, 0, 0, 0, 1, 0, 0, 0, 0,

            0, 0, 0, 0, 1, 0, 0, 0, 0,

            0, 0, 0, 0, 1, 0, 0, 0, 0;

        for (index_t i = 0; i < numCases; i++) {
            WHEN("An axis-aligned ray passes through the center of a pixel")
            {
                INFO("A ", al[i], "-axis-aligned ray passes through the center of a pixel");

                geom.emplace_back(stc, ctr, std::move(volData), std::move(sinoData),
                                  RotationAngles3D{Gamma{gamma[i]}, Beta{beta[i]}});

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer sino(sinoDescriptor);

                JosephsMethod op(volumeDescriptor, sinoDescriptor,
                                 JosephsMethod<>::Interpolation::LINEAR);

                THEN("The result of projecting through a voxel is exactly the voxel value")
                {
                    for (index_t j = 0; j < volSize; j++) {
                        volume = 0;
                        if (i / 2 == 0)
                            volume(volSize / 2, volSize / 2, j) = 1;
                        else if (i / 2 == 1)
                            volume(j, volSize / 2, volSize / 2) = 1;
                        else if (i / 2 == 2)
                            volume(volSize / 2, j, volSize / 2) = 1;

                        op.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx(1.0));
                    }

                    AND_THEN("The backprojection sets the values of all hit voxels to the detector "
                             "value")
                    {
                        op.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(
                            isApprox(volume, DataContainer(volumeDescriptor, backProj[i / 2])));
                    }
                }
            }
        }

        real_t offsetx[numCases] = {0.25, 0.25, 0.0, 0.0, 0.0, 0.0};
        real_t offsety[numCases] = {0.0, 0.0, 0.25, 0.25, 0.0, 0.0};
        real_t offsetz[numCases] = {0.0, 0.0, 0.0, 0.0, 0.25, 0.25};

        backProj[2] << 0, 0.25, 0, 0, 0.25, 0, 0, 0.25, 0,

            0, 0.75, 0, 0, 0.75, 0, 0, 0.75, 0,

            0, 0, 0, 0, 0, 0, 0, 0, 0;

        backProj[1] << 0, 0, 0, 0, 0, 0, 0, 0, 0,

            0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0, 0, 0,

            0, 0, 0, 0, 0, 0, 0, 0, 0;

        backProj[0] << 0, 0, 0, 0.25, 0.75, 0, 0, 0, 0,

            0, 0, 0, 0.25, 0.75, 0, 0, 0, 0,

            0, 0, 0, 0.25, 0.75, 0, 0, 0, 0;

        for (index_t i = 0; i < numCases; i++) {
            WHEN("An axis-aligned ray does not pass through the center of a voxel")
            {
                INFO("A ", al[i], "-axis-aligned ray does not pass through the center of a voxel");

                geom.emplace_back(SourceToCenterOfRotation{volSize * 2000}, ctr, std::move(volData),
                                  std::move(sinoData),
                                  RotationAngles3D{Gamma{gamma[i]}, Beta{beta[i]}},
                                  PrincipalPointOffset2D{0, 0},
                                  RotationOffset3D{-offsetx[i], -offsety[i], -offsetz[i]});

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer sino(sinoDescriptor);

                JosephsMethod op(volumeDescriptor, sinoDescriptor,
                                 JosephsMethod<>::Interpolation::LINEAR);

                THEN("The result of projecting through a voxel is the interpolated value between "
                     "the four voxels nearest to the ray")
                {
                    for (index_t j = 0; j < volSize; j++) {
                        volume = 0;
                        if (i / 2 == 0)
                            volume(volSize / 2, volSize / 2, j) = 1;
                        else if (i / 2 == 1)
                            volume(j, volSize / 2, volSize / 2) = 1;
                        else if (i / 2 == 2)
                            volume(volSize / 2, j, volSize / 2) = 1;

                        op.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx(0.75));
                    }

                    AND_THEN("The backprojection yields the exact adjoint")
                    {
                        sino[0] = 1;

                        op.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(
                            isApprox(volume, DataContainer(volumeDescriptor, backProj[i / 2])));
                    }
                }
            }
        }

        offsetx[0] = volSize / 2.0;
        offsetx[1] = -(volSize / 2.0);
        offsetx[2] = 0.0;
        offsetx[3] = 0.0;
        offsetx[4] = -(volSize / 2.0);
        offsetx[5] = (volSize / 2.0);

        offsety[0] = 0.0;
        offsety[1] = 0.0;
        offsety[2] = volSize / 2.0;
        offsety[3] = -(volSize / 2.0);
        offsety[4] = -(volSize / 2.0);
        offsety[5] = (volSize / 2.0);

        al[0] = "left border";
        al[1] = "right border";
        al[2] = "bottom border";
        al[3] = "top border";
        al[4] = "top right edge";
        al[5] = "bottom left edge";

        backProj[0] << 0, 0, 0, 1, 0, 0, 0, 0, 0,

            0, 0, 0, 1, 0, 0, 0, 0, 0,

            0, 0, 0, 1, 0, 0, 0, 0, 0;

        backProj[1] << 0, 0, 0, 0, 0, 1, 0, 0, 0,

            0, 0, 0, 0, 0, 1, 0, 0, 0,

            0, 0, 0, 0, 0, 1, 0, 0, 0;

        backProj[2] << 0, 1, 0, 0, 0, 0, 0, 0, 0,

            0, 1, 0, 0, 0, 0, 0, 0, 0,

            0, 1, 0, 0, 0, 0, 0, 0, 0;

        backProj[3] << 0, 0, 0, 0, 0, 0, 0, 1, 0,

            0, 0, 0, 0, 0, 0, 0, 1, 0,

            0, 0, 0, 0, 0, 0, 0, 1, 0;

        backProj[4] << 0, 0, 0, 0, 0, 0, 0, 0, 1,

            0, 0, 0, 0, 0, 0, 0, 0, 1,

            0, 0, 0, 0, 0, 0, 0, 0, 1;

        backProj[5] << 1, 0, 0, 0, 0, 0, 0, 0, 0,

            1, 0, 0, 0, 0, 0, 0, 0, 0,

            1, 0, 0, 0, 0, 0, 0, 0, 0;

        for (index_t i = 0; i < numCases; i++) {
            WHEN("A z-axis-aligned ray runs along the a corner of the volume")
            {
                INFO("A z-axis-aligned ray runs along the ", al[i], " of the volume");
                // x-ray source must be very far from the volume center to make testing of the fast
                // backprojection simpler
                geom.emplace_back(SourceToCenterOfRotation{volSize * 2000}, ctr, std::move(volData),
                                  std::move(sinoData), RotationAngles3D{Gamma{0}},
                                  PrincipalPointOffset2D{0, 0},
                                  RotationOffset3D{-offsetx[i], -offsety[i], 0});

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer sino(sinoDescriptor);

                JosephsMethod op(volumeDescriptor, sinoDescriptor);

                THEN("The result of projecting through a voxel is exactly the voxel's value (we "
                     "mirror values at the border for the purpose of interpolation)")
                {
                    for (index_t j = 0; j < volSize; j++) {
                        volume = 0;
                        switch (i) {
                            case 0:
                                volume(0, volSize / 2, j) = 1;
                                break;
                            case 1:
                                volume(volSize - 1, volSize / 2, j) = 1;
                                break;
                            case 2:
                                volume(volSize / 2, 0, j) = 1;
                                break;
                            case 3:
                                volume(volSize / 2, volSize - 1, j) = 1;
                                break;
                            case 4:
                                volume(volSize - 1, volSize - 1, j) = 1;
                                break;
                            case 5:
                                volume(0, 0, j) = 1;
                                break;
                            default:
                                break;
                        }

                        op.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx(1));
                    }

                    AND_THEN("The backprojection yields the exact adjoint")
                    {
                        sino[0] = 1;

                        op.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(
                            isApprox(volume, DataContainer(volumeDescriptor, backProj[i])));
                    }
                }
            }
        }
    }

    GIVEN("A 3D setting with multiple projection angles")
    {
        IndexVector_t volumeDims(3), sinoDims(3);
        const index_t volSize = 3;
        const index_t detectorSize = 1;
        const index_t numImgs = 6;
        volumeDims << volSize, volSize, volSize;
        sinoDims << detectorSize, detectorSize, numImgs;
        VolumeDescriptor volumeDescriptor(volumeDims);
        DataContainer volume(volumeDescriptor);

        auto stc = SourceToCenterOfRotation{20 * volSize};
        auto ctr = CenterOfRotationToDetector{volSize};
        auto volData = VolumeData3D{Size3D{volumeDims}};
        auto sinoData = SinogramData3D{Size3D{sinoDims}};

        std::vector<Geometry> geom;

        WHEN("x-, y and z-axis-aligned rays are present")
        {
            real_t beta[numImgs] = {0.0, 0.0, 0.0, 0.0, pi_t / 2, 3 * pi_t / 2};
            real_t gamma[numImgs] = {0.0, pi_t, pi_t / 2, 3 * pi_t / 2, pi_t / 2, 3 * pi_t / 2};

            for (index_t i = 0; i < numImgs; i++)
                geom.emplace_back(stc, ctr, VolumeData3D{Size3D{volumeDims}},
                                  SinogramData3D{Size3D{sinoDims}},
                                  RotationAngles3D{Gamma{gamma[i]}, Beta{beta[i]}});

            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);

            JosephsMethod op(volumeDescriptor, sinoDescriptor,
                             JosephsMethod<>::Interpolation::LINEAR);

            THEN("Values are accumulated correctly along each ray's path")
            {
                volume = 0;

                // set only values along the rays' path to one to make sure interpolation is done
                // correctly
                for (index_t i = 0; i < volSize; i++) {
                    volume(i, volSize / 2, volSize / 2) = 1;
                    volume(volSize / 2, i, volSize / 2) = 1;
                    volume(volSize / 2, volSize / 2, i) = 1;
                }

                op.apply(volume, sino);
                for (index_t i = 0; i < numImgs; i++)
                    REQUIRE_EQ(sino[i], Approx(3.0));

                AND_THEN("Backprojections yield the exact adjoint")
                {
                    RealVector_t cmp(volSize * volSize * volSize);

                    cmp << 0, 0, 0, 0, 6, 0, 0, 0, 0,

                        0, 6, 0, 6, 18, 6, 0, 6, 0,

                        0, 0, 0, 0, 6, 0, 0, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isApprox(volume, DataContainer(volumeDescriptor, cmp)));
                }
            }
        }
    }

    GIVEN("A non-cubic 3D volume")
    {
        IndexVector_t volumeDims(3), sinoDims(3);
        volumeDims << 5, 1, 2;
        const index_t detectorSize = 1;
        const index_t numImgs = 1;
        sinoDims << detectorSize, detectorSize, numImgs;
        VolumeDescriptor volumeDescriptor(volumeDims);
        DataContainer volume(volumeDescriptor);

        auto stc = SourceToCenterOfRotation{20 * 5};
        auto ctr = CenterOfRotationToDetector{5};
        auto volData = VolumeData3D{Size3D{volumeDims}};
        auto sinoData = SinogramData3D{Size3D{sinoDims}};

        std::vector<Geometry> geom;

        WHEN("An axis-aligned ray enters (and leaves) through the shorter volume dimension")
        {
            geom.emplace_back(stc, ctr, std::move(volData), std::move(sinoData),
                              RotationAngles3D{Gamma{0}}, PrincipalPointOffset2D{0, 0},
                              RotationOffset3D{static_cast<real_t>(1.6), 0, 0});

            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);

            JosephsMethod op(volumeDescriptor, sinoDescriptor);

            THEN("The results of forward projecting are correct")
            {
                volume = 0;
                volume(4, 0, 0) = 1;
                volume(4, 0, 1) = 1;

                op.apply(volume, sino);
                REQUIRE_EQ(sino[0], Approx(1.2));

                AND_THEN("The backprojection yields the correct result")
                {
                    sino[0] = 1;

                    volume = 0;
                    DataContainer bpExpected = volume;
                    bpExpected(4, 0, 0) = static_cast<real_t>(0.6);
                    bpExpected(4, 0, 1) = static_cast<real_t>(0.6);
                    bpExpected(3, 0, 0) = static_cast<real_t>(0.4);
                    bpExpected(3, 0, 1) = static_cast<real_t>(0.4);

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isApprox(volume, bpExpected));
                }
            }
        }
    }
}

TEST_CASE("JosephsMethod: Projection under an angle")
{
    // Turn logger of
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("A 2D setting with a single ray")
    {
        IndexVector_t volumeDims(2), sinoDims(2);
        const index_t volSize = 4;
        const index_t detectorSize = 1;
        const index_t numImgs = 1;
        volumeDims << volSize, volSize;
        sinoDims << detectorSize, numImgs;
        VolumeDescriptor volumeDescriptor(volumeDims);
        DataContainer volume(volumeDescriptor);

        auto stc = SourceToCenterOfRotation{20 * volSize};
        auto ctr = CenterOfRotationToDetector{volSize};
        auto volData = VolumeData2D{Size2D{volumeDims}};
        auto sinoData = SinogramData2D{Size2D{sinoDims}};

        std::vector<Geometry> geom;

        WHEN("Projecting under an angle of 30 degrees and ray goes through center of volume")
        {
            // In this case the ray enters and exits the volume through the borders along the main
            // direction Weighting for all interpolated values should be the same
            geom.emplace_back(stc, ctr, Radian{-pi_t / 6}, std::move(volData), std::move(sinoData));

            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);

            JosephsMethod op(volumeDescriptor, sinoDescriptor,
                             JosephsMethod<>::Interpolation::LINEAR);

            real_t weight = static_cast<real_t>(2 / std::sqrt(3.f));
            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                volume(3, 0) = 0;
                volume(1, 1) = 0;
                volume(1, 2) = 0;
                volume(1, 3) = 0;

                volume(2, 0) = 0;
                volume(2, 1) = 0;
                volume(2, 2) = 0;
                volume(0, 3) = 0;

                op.apply(volume, sino);
                DataContainer zero(sinoDescriptor);
                zero = 0;
                CHECK_UNARY(isApprox(sino, zero, epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    volume(3, 0) = 1;
                    volume(1, 1) = 1;
                    volume(1, 2) = 1;
                    volume(1, 3) = 1;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0], Approx(2 * weight));

                    sino[0] = 1;

                    RealVector_t opExpected(volSize * volSize);
                    opExpected << 0, 0, (3 - std::sqrt(3.f)) / 2, (std::sqrt(3.f) - 1) / 2, 0,
                        (std::sqrt(3.f) - 1) / (2 * std::sqrt(3.f)),
                        (std::sqrt(3.f) + 1) / (2 * std::sqrt(3.f)), 0, 0,
                        (std::sqrt(3.f) + 1) / (2 * std::sqrt(3.f)),
                        (std::sqrt(3.f) - 1) / (2 * std::sqrt(3.f)), 0, (std::sqrt(3.f) - 1) / 2,
                        (3 - std::sqrt(3.f)) / 2, 0, 0;

                    opExpected *= weight;
                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isApprox(volume, DataContainer(volumeDescriptor, opExpected)));
                }
            }
        }

        WHEN("Projecting under an angle of 30 degrees and ray enters through the right border")
        {
            // In this case the ray exits through a border along the main ray direction, but enters
            // through a border not along the main direction First pixel should be weighted
            // differently
            geom.emplace_back(stc, ctr, Radian{-pi_t / 6}, std::move(volData), std::move(sinoData),
                              PrincipalPointOffset{0}, RotationOffset2D{std::sqrt(3.f), 0});

            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);

            JosephsMethod op(volumeDescriptor, sinoDescriptor,
                             JosephsMethod<>::Interpolation::LINEAR);

            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                volume(3, 1) = 0;
                volume(3, 2) = 0;
                volume(3, 3) = 0;
                volume(2, 3) = 0;
                volume(2, 2) = 0;

                op.apply(volume, sino);
                DataContainer zero(sinoDescriptor);
                zero = 0;
                CHECK_UNARY(isApprox(sino, zero, epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    volume(3, 1) = 1;
                    volume(2, 2) = 1;
                    volume(2, 3) = 1;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0],
                               Approx((4 - 2 * std::sqrt(3.f)) * (std::sqrt(3.f) - 1)
                                      + (2 / std::sqrt(3.f)) * (3 - 8 * std::sqrt(3.f) / 6))
                                   .epsilon(epsilon));

                    sino[0] = 1;

                    RealVector_t opExpected(volSize * volSize);
                    opExpected << 0, 0, 0, 0, 0, 0, 0,
                        (4 - 2 * std::sqrt(3.f)) * (std::sqrt(3.f) - 1), 0, 0,
                        static_cast<real_t>(2 / std::sqrt(3.f))
                            * static_cast<real_t>(1.5 - 5 * std::sqrt(3.f) / 6),
                        (4 - 2 * std::sqrt(3.f)) * (2 - std::sqrt(3.f))
                            + (2 / std::sqrt(3.f)) * (5 * std::sqrt(3.f) / 6 - 0.5f),
                        0, 0, (2 / std::sqrt(3.f)) * (1.5f - std::sqrt(3.f) / 2),
                        (2 / std::sqrt(3.f)) * (std::sqrt(3.f) / 2 - 0.5f);

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(
                        isApprox(volume, DataContainer(volumeDescriptor, opExpected), epsilon));
                }
            }
        }

        WHEN("Projecting under an angle of 30 degrees and ray exits through the left border")
        {
            // In this case the ray enters through a border along the main ray direction, but exits
            // through a border not along the main direction Last pixel should be weighted
            // differently
            geom.emplace_back(stc, ctr, Radian{-pi_t / 6}, std::move(volData), std::move(sinoData),
                              PrincipalPointOffset{0}, RotationOffset2D{-std::sqrt(3.f), 0});

            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);

            JosephsMethod op(volumeDescriptor, sinoDescriptor,
                             JosephsMethod<>::Interpolation::LINEAR);

            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                volume(0, 0) = 0;
                volume(1, 0) = 0;
                volume(0, 1) = 0;
                volume(1, 1) = 0;
                volume(0, 2) = 0;

                op.apply(volume, sino);
                DataContainer zero(sinoDescriptor);
                zero = 0;
                CHECK_UNARY(isApprox(sino, zero, epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    volume(1, 0) = 1;
                    volume(0, 1) = 1;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0],
                               Approx((std::sqrt(3.f) - 1) + (5.0 / 3.0 - 1 / std::sqrt(3.f))
                                      + (4 - 2 * std::sqrt(3.f)) * (2 - std::sqrt(3.f)))
                                   .epsilon(epsilon));

                    sino[0] = 1;

                    RealVector_t opExpected(volSize * volSize);
                    opExpected << 1 - 1 / std::sqrt(3.f), std::sqrt(3.f) - 1, 0, 0,
                        (5.0f / 3.0f - 1 / std::sqrt(3.f))
                            + (4 - 2 * std::sqrt(3.f)) * (2 - std::sqrt(3.f)),
                        std::sqrt(3.f) - 5.0f / 3.0f, 0, 0,
                        (std::sqrt(3.f) - 1) * (4 - 2 * std::sqrt(3.f)), 0, 0, 0, 0, 0, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isApprox(volume, DataContainer(volumeDescriptor, opExpected)));
                }
            }
        }

        WHEN("Projecting under an angle of 30 degrees and ray only intersects a single pixel")
        {
            // This is a special case that is handled separately in both forward and backprojection
            geom.emplace_back(stc, ctr, Radian{-pi_t / 6}, std::move(volData), std::move(sinoData),
                              PrincipalPointOffset{0},
                              RotationOffset2D{-2 - std::sqrt(3.f) / 2, 0});

            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);

            JosephsMethod op(volumeDescriptor, sinoDescriptor,
                             JosephsMethod<>::Interpolation::LINEAR);

            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                volume(0, 0) = 0;

                op.apply(volume, sino);
                DataContainer zero(sinoDescriptor);
                zero = 0;
                CHECK_UNARY(isApprox(sino, zero, epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    volume(0, 0) = 1;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0], Approx(1 / std::sqrt(3.f)).epsilon(epsilon));

                    sino[0] = 1;

                    RealVector_t opExpected(volSize * volSize);
                    opExpected << 1 / std::sqrt(3.f), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isApprox(volume, DataContainer(volumeDescriptor, opExpected)));
                }
            }
        }

        WHEN("Projecting under an angle of 120 degrees and ray goes through center of volume")
        {
            // In this case the ray enters and exits the volume through the borders along the main
            // direction Weighting for all interpolated values should be the same
            geom.emplace_back(stc, ctr, Radian{-2 * pi_t / 3}, std::move(volData),
                              std::move(sinoData));

            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);

            JosephsMethod op(volumeDescriptor, sinoDescriptor,
                             JosephsMethod<>::Interpolation::LINEAR);

            real_t weight = 2 / std::sqrt(3.f);
            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                volume(0, 0) = 0;
                volume(0, 1) = 0;
                volume(1, 1) = 0;
                volume(1, 2) = 0;

                volume(2, 1) = 0;
                volume(2, 2) = 0;
                volume(3, 2) = 0;
                volume(3, 3) = 0;

                op.apply(volume, sino);
                DataContainer zero(sinoDescriptor);
                zero = 0;
                CHECK_UNARY(isApprox(sino, zero, epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    volume(3, 3) = 1;
                    volume(0, 1) = 1;
                    volume(1, 1) = 1;
                    volume(2, 1) = 1;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0], Approx(2 * weight));

                    sino[0] = 1;

                    RealVector_t opExpected(volSize * volSize);

                    opExpected << (std::sqrt(3.f) - 1) / 2, 0, 0, 0, (3 - std::sqrt(3.f)) / 2,
                        (std::sqrt(3.f) + 1) / (2 * std::sqrt(3.f)),
                        (std::sqrt(3.f) - 1) / (2 * std::sqrt(3.f)), 0, 0,
                        (std::sqrt(3.f) - 1) / (2 * std::sqrt(3.f)),
                        (std::sqrt(3.f) + 1) / (2 * std::sqrt(3.f)), (3 - std::sqrt(3.f)) / 2, 0, 0,
                        0, (std::sqrt(3.f) - 1) / 2;

                    opExpected *= weight;
                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isApprox(volume, DataContainer(volumeDescriptor, opExpected)));
                }
            }
        }

        WHEN("Projecting under an angle of 120 degrees and ray enters through the top border")
        {
            // In this case the ray exits through a border along the main ray direction, but enters
            // through a border not along the main direction First pixel should be weighted
            // differently
            geom.emplace_back(stc, ctr, Radian{-2 * pi_t / 3}, std::move(volData),
                              std::move(sinoData), PrincipalPointOffset{0},
                              RotationOffset2D{0, std::sqrt(3.f)});

            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);

            JosephsMethod op(volumeDescriptor, sinoDescriptor,
                             JosephsMethod<>::Interpolation::LINEAR);

            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                volume(0, 2) = 0;
                volume(0, 3) = 0;
                volume(1, 2) = 0;
                volume(1, 3) = 0;
                volume(2, 3) = 0;

                op.apply(volume, sino);
                DataContainer zero(sinoDescriptor);
                zero = 0;
                CHECK_UNARY(isApprox(sino, zero, epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    volume(2, 3) = 1;
                    volume(1, 2) = 1;
                    volume(1, 3) = 1;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0], Approx((4 - 2 * std::sqrt(3.f)) + (2 / std::sqrt(3.f))));

                    sino[0] = 1;

                    RealVector_t opExpected(volSize * volSize);

                    opExpected << 0, 0, 0, 0, 0, 0, 0, 0,
                        (2 / std::sqrt(3.f)) * (1.5f - std::sqrt(3.f) / 2),
                        (2 / std::sqrt(3.f)) * (1.5f - 5 * std::sqrt(3.f) / 6), 0, 0,
                        (2 / std::sqrt(3.f)) * (std::sqrt(3.f) / 2 - 0.5f),
                        (4 - 2 * std::sqrt(3.f)) * (2 - std::sqrt(3.f))
                            + (2 / std::sqrt(3.f)) * (5 * std::sqrt(3.f) / 6 - 0.5f),
                        (4 - 2 * std::sqrt(3.f)) * (std::sqrt(3.f) - 1), 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(
                        isApprox(volume, DataContainer(volumeDescriptor, opExpected), epsilon));
                }
            }
        }

        WHEN("Projecting under an angle of 120 degrees and ray exits through the bottom border")
        {
            // In this case the ray enters through a border along the main ray direction, but exits
            // through a border not along the main direction Last pixel should be weighted
            // differently
            geom.emplace_back(stc, ctr, Radian{-2 * pi_t / 3}, std::move(volData),
                              std::move(sinoData), PrincipalPointOffset{0},
                              RotationOffset2D{0, -std::sqrt(3.f)});

            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);

            JosephsMethod op(volumeDescriptor, sinoDescriptor,
                             JosephsMethod<>::Interpolation::LINEAR);

            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                volume(1, 0) = 0;
                volume(2, 0) = 0;
                volume(3, 0) = 0;
                volume(2, 1) = 0;
                volume(3, 1) = 0;

                op.apply(volume, sino);
                DataContainer zero(sinoDescriptor);
                zero = 0;
                CHECK_UNARY(isApprox(sino, zero, epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    volume(2, 0) = 1;
                    volume(3, 1) = 1;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0],
                               Approx((std::sqrt(3.f) - 1) + (5.0 / 3.0 - 1 / std::sqrt(3.f))
                                      + (4 - 2 * std::sqrt(3.f)) * (2 - std::sqrt(3.f)))
                                   .epsilon(epsilon));

                    sino[0] = 1;

                    RealVector_t opExpected(volSize * volSize);

                    opExpected << 0, (std::sqrt(3.f) - 1) * (4 - 2 * std::sqrt(3.f)),
                        (5.0f / 3.0f - 1 / std::sqrt(3.f))
                            + (4 - 2 * std::sqrt(3.f)) * (2 - std::sqrt(3.f)),
                        1 - 1 / std::sqrt(3.f), 0, 0, std::sqrt(3.f) - 5.0f / 3.0f,
                        std::sqrt(3.f) - 1, 0, 0, 0, 0, 0, 0, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isApprox(volume, DataContainer(volumeDescriptor, opExpected)));
                }
            }
        }

        WHEN("Projecting under an angle of 120 degrees and ray only intersects a single pixel")
        {
            // This is a special case that is handled separately in both forward and backprojection
            geom.emplace_back(stc, ctr, Radian{-2 * pi_t / 3}, std::move(volData),
                              std::move(sinoData), PrincipalPointOffset{0},
                              RotationOffset2D{0, -2 - std::sqrt(3.f) / 2});

            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);

            JosephsMethod op(volumeDescriptor, sinoDescriptor,
                             JosephsMethod<>::Interpolation::LINEAR);

            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                volume(3, 0) = 0;

                op.apply(volume, sino);
                DataContainer zero(sinoDescriptor);
                zero = 0;
                CHECK_UNARY(isApprox(sino, zero, epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    volume(3, 0) = 1;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0], Approx(1 / std::sqrt(3.f)).epsilon(epsilon));

                    sino[0] = 1;

                    RealVector_t opExpected(volSize * volSize);
                    opExpected << 0, 0, 0, 1 / std::sqrt(3.f), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(
                        isApprox(volume, DataContainer(volumeDescriptor, opExpected), epsilon));
                }
            }
        }
    }

    GIVEN("A non-quadratic 2D volume")
    {
        IndexVector_t volumeDims(2), sinoDims(2);
        volumeDims << 4, 1;
        const index_t detectorSize = 1;
        const index_t numImgs = 1;
        sinoDims << detectorSize, numImgs;
        VolumeDescriptor volumeDescriptor(volumeDims);
        DataContainer volume(volumeDescriptor);

        auto stc = SourceToCenterOfRotation{20 * 5};
        auto ctr = CenterOfRotationToDetector{5};
        auto volData = VolumeData2D{Size2D{volumeDims}};
        auto sinoData = SinogramData2D{Size2D{sinoDims}};

        std::vector<Geometry> geom;

        WHEN("An axis-aligned ray enters (and leaves) through the shorter volume dimension")
        {
            geom.emplace_back(stc, ctr, Radian{pi_t / 6}, std::move(volData), std::move(sinoData));

            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);

            JosephsMethod op(volumeDescriptor, sinoDescriptor);

            THEN("The results of forward projecting are correct")
            {
                volume = 0;
                volume(1, 0) = 1;

                op.apply(volume, sino);
                REQUIRE_EQ(sino[0], Approx(std::sqrt(static_cast<real_t>(1) / 3)));

                AND_THEN("The backprojection yields the correct result")
                {
                    sino[0] = 1;

                    volume = 0;
                    DataContainer bpExpected = volume;
                    bpExpected(1, 0) = std::sqrt(static_cast<real_t>(1) / 3);
                    bpExpected(2, 0) = std::sqrt(static_cast<real_t>(1) / 3);

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isApprox(volume, bpExpected));
                }
            }
        }
    }

    GIVEN("A 3D setting with a single ray")
    {
        IndexVector_t volumeDims(3), sinoDims(3);
        const index_t volSize = 3;
        const index_t detectorSize = 1;
        const index_t numImgs = 1;
        volumeDims << volSize, volSize, volSize;
        sinoDims << detectorSize, detectorSize, numImgs;
        VolumeDescriptor volumeDescriptor(volumeDims);
        DataContainer volume(volumeDescriptor);

        auto stc = SourceToCenterOfRotation{20 * volSize};
        auto ctr = CenterOfRotationToDetector{volSize};
        auto volData = VolumeData3D{Size3D{volumeDims}};
        auto sinoData = SinogramData3D{Size3D{sinoDims}};

        std::vector<Geometry> geom;

        RealVector_t backProj(volSize * volSize * volSize);

        WHEN("A ray with an angle of 30 degrees goes through the center of the volume")
        {
            // In this case the ray enters and exits the volume along the main direction
            geom.emplace_back(stc, ctr, std::move(volData), std::move(sinoData),
                              RotationAngles3D{Gamma{pi_t / 6}});

            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);

            JosephsMethod op(volumeDescriptor, sinoDescriptor,
                             JosephsMethod<>::Interpolation::LINEAR);

            THEN("The ray intersects the correct voxels")
            {
                volume = 1;
                volume(1, 1, 1) = 0;
                volume(2, 1, 0) = 0;
                volume(1, 1, 0) = 0;
                volume(0, 1, 2) = 0;
                volume(1, 1, 2) = 0;

                op.apply(volume, sino);
                REQUIRE_EQ(sino[0], Approx(0).epsilon(1e-5));

                AND_THEN("The correct weighting is applied")
                {
                    volume(1, 1, 1) = 1;
                    volume(2, 1, 0) = 3;
                    volume(1, 1, 2) = 2;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0],
                               Approx(2 / std::sqrt(3.f) + 2 - 4.0f / 3 + 4 / std::sqrt(3.f)));

                    sino[0] = 1;
                    backProj << 0, 0, 0, 0, 2 / std::sqrt(3.f) - 2.0f / 3, 2.0f / 3, 0, 0, 0,

                        0, 0, 0, 0, 2 / std::sqrt(3.f), 0, 0, 0, 0,

                        0, 0, 0, 2.0f / 3, 2 / std::sqrt(3.f) - 2.0f / 3, 0, 0, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isApprox(volume, DataContainer(volumeDescriptor, backProj)));
                }
            }
        }

        WHEN("A ray with an angle of 30 degrees enters through the right border")
        {
            // In this case the ray enters through a border orthogonal to a non-main direction
            geom.emplace_back(stc, ctr, std::move(volData), std::move(sinoData),
                              RotationAngles3D{Gamma{pi_t / 6}}, PrincipalPointOffset2D{0, 0},
                              RotationOffset3D{1, 0, 0});

            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);

            JosephsMethod op(volumeDescriptor, sinoDescriptor,
                             JosephsMethod<>::Interpolation::LINEAR);

            THEN("The ray intersects the correct voxels")
            {
                volume = 1;
                volume(2, 1, 1) = 0;
                volume(2, 1, 0) = 0;
                volume(2, 1, 2) = 0;
                volume(1, 1, 2) = 0;

                op.apply(volume, sino);
                REQUIRE_EQ(sino[0], Approx(0).epsilon(1e-5));

                AND_THEN("The correct weighting is applied")
                {
                    volume(2, 1, 0) = 4;
                    volume(1, 1, 2) = 3;
                    volume(2, 1, 1) = 1;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0], Approx((std::sqrt(3.f) + 1) * (1 - 1 / std::sqrt(3.f)) + 3
                                               - std::sqrt(3.f) / 2 + 2 / std::sqrt(3.f))
                                            .epsilon(epsilon));

                    sino[0] = 1;
                    backProj << 0, 0, 0, 0, 0,
                        ((std::sqrt(3.f) + 1) / 4) * (1 - 1 / std::sqrt(3.f)), 0, 0, 0,

                        0, 0, 0, 0, 0, 2 / std::sqrt(3.f) + 1 - std::sqrt(3.f) / 2, 0, 0, 0,

                        0, 0, 0, 0, 2.0f / 3, 2 / std::sqrt(3.f) - 2.0f / 3, 0, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isApprox(volume, DataContainer(volumeDescriptor, backProj)));
                }
            }
        }

        WHEN("A ray with an angle of 30 degrees exits through the left border")
        {
            // In this case the ray exit through a border orthogonal to a non-main direction
            geom.emplace_back(stc, ctr, std::move(volData), std::move(sinoData),
                              RotationAngles3D{Gamma{pi_t / 6}}, PrincipalPointOffset2D{0, 0},
                              RotationOffset3D{-1, 0, 0});

            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);

            JosephsMethod op(volumeDescriptor, sinoDescriptor,
                             JosephsMethod<>::Interpolation::LINEAR);

            THEN("The ray intersects the correct voxels")
            {
                volume = 1;
                volume(0, 1, 0) = 0;
                volume(1, 1, 0) = 0;
                volume(0, 1, 1) = 0;
                volume(0, 1, 2) = 0;

                op.apply(volume, sino);
                REQUIRE_EQ(sino[0], Approx(0).epsilon(1e-5));

                AND_THEN("The correct weighting is applied")
                {
                    volume(0, 1, 2) = 4;
                    volume(1, 1, 0) = 3;
                    volume(0, 1, 1) = 1;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0], Approx((std::sqrt(3.f) + 1) * (1 - 1 / std::sqrt(3.f)) + 3
                                               - std::sqrt(3.f) / 2 + 2 / std::sqrt(3.f))
                                            .epsilon(epsilon));

                    sino[0] = 1;
                    backProj << 0, 0, 0, 2 / std::sqrt(3.f) - 2.0f / 3, 2.0f / 3, 0, 0, 0, 0,

                        0, 0, 0, 2 / std::sqrt(3.f) + 1 - std::sqrt(3.f) / 2, 0, 0, 0, 0, 0,

                        0, 0, 0, ((std::sqrt(3.f) + 1) / 4) * (1 - 1 / std::sqrt(3.f)), 0, 0, 0, 0,
                        0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isApprox(volume, DataContainer(volumeDescriptor, backProj)));
                }
            }
        }

        WHEN("A ray with an angle of 30 degrees only intersects a single voxel")
        {
            // special case - no interior voxels, entry and exit voxels are the same
            geom.emplace_back(stc, ctr, std::move(volData), std::move(sinoData),
                              RotationAngles3D{Gamma{pi_t / 6}}, PrincipalPointOffset2D{0, 0},
                              RotationOffset3D{-2, 0, 0});

            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);

            JosephsMethod op(volumeDescriptor, sinoDescriptor,
                             JosephsMethod<>::Interpolation::LINEAR);

            THEN("The ray intersects the correct voxels")
            {
                volume = 1;
                volume(0, 1, 0) = 0;

                op.apply(volume, sino);
                REQUIRE_EQ(sino[0], Approx(0).epsilon(1e-5));

                AND_THEN("The correct weighting is applied")
                {
                    volume(0, 1, 0) = 1;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0], Approx(std::sqrt(3.f) - 1).epsilon(epsilon));

                    sino[0] = 1;
                    backProj << 0, 0, 0, std::sqrt(3.f) - 1, 0, 0, 0, 0, 0,

                        0, 0, 0, 0, 0, 0, 0, 0, 0,

                        0, 0, 0, 0, 0, 0, 0, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(
                        isApprox(volume, DataContainer(volumeDescriptor, backProj), epsilon));
                }
            }
        }
    }

    GIVEN("A non-cubic 3D volume")
    {
        IndexVector_t volumeDims(3), sinoDims(3);
        volumeDims << 4, 1, 1;
        const index_t detectorSize = 1;
        const index_t numImgs = 1;
        sinoDims << detectorSize, detectorSize, numImgs;
        VolumeDescriptor volumeDescriptor(volumeDims);
        DataContainer volume(volumeDescriptor);

        auto stc = SourceToCenterOfRotation{20 * 5};
        auto ctr = CenterOfRotationToDetector{5};
        auto volData = VolumeData3D{Size3D{volumeDims}};
        auto sinoData = SinogramData3D{Size3D{sinoDims}};

        std::vector<Geometry> geom;

        WHEN("An axis-aligned ray enters (and leaves) through the shorter volume dimension")
        {
            geom.emplace_back(stc, ctr, std::move(volData), std::move(sinoData),
                              RotationAngles3D{Gamma{pi_t / 6}});

            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);

            JosephsMethod op(volumeDescriptor, sinoDescriptor);

            THEN("The results of forward projecting are correct")
            {
                volume = 0;
                volume(1, 0, 0) = 1;

                op.apply(volume, sino);
                REQUIRE_EQ(sino[0], Approx(std::sqrt(static_cast<real_t>(1) / 3)));

                AND_THEN("The backprojection yields the correct result")
                {
                    sino[0] = 1;

                    volume = 0;
                    DataContainer bpExpected = volume;
                    bpExpected(1, 0, 0) = std::sqrt(static_cast<real_t>(1) / 3);
                    bpExpected(2, 0, 0) = std::sqrt(static_cast<real_t>(1) / 3);

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isApprox(volume, bpExpected, epsilon));
                }
            }
        }
    }
}
