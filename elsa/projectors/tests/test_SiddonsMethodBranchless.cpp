#include "doctest/doctest.h"

#include "SiddonsMethodBranchless.h"
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

TEST_CASE("SiddonMethod: Testing projector with only one ray")
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

    GIVEN("A SiddonsMethod for 2D and a domain data with all ones")
    {
        std::vector<Geometry> geom;

        auto dataDomain = DataContainer(domain);
        dataDomain = 1;

        WHEN("We have a single ray with 0 degrees")
        {
            VolumeData2D volDataCopy{volData};
            SinogramData2D sinoDataCopy{sinoData};
            geom.emplace_back(stc, ctr, Radian{0}, std::move(volDataCopy), std::move(sinoDataCopy));

            auto detectorDesc = PlanarDetectorDescriptor(sizeRange, {geom});
            auto dataRange = DataContainer(detectorDesc);
            dataRange = 0;

            auto op = SiddonsMethodBranchless(domain, detectorDesc);

            THEN("A^t A x should be close to the original data")
            {
                auto AtAx = DataContainer(domain);

                op.apply(dataDomain, dataRange);
                op.applyAdjoint(dataRange, AtAx);

                auto cmp = RealVector_t(sizeDomain.prod());
                cmp << 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0;

                REQUIRE_UNARY(isCwiseApprox(DataContainer(domain, cmp), AtAx));
                REQUIRE_EQ(dataRange[0], Approx(5));
            }
        }

        WHEN("We have a single ray with 180 degrees")
        {
            VolumeData2D volDataCopy{volData};
            SinogramData2D sinoDataCopy{sinoData};
            geom.emplace_back(stc, ctr, Degree{180}, std::move(volDataCopy),
                              std::move(sinoDataCopy));

            auto detectorDesc = PlanarDetectorDescriptor(sizeRange, {geom});
            auto dataRange = DataContainer(detectorDesc);
            dataRange = 0;

            auto op = SiddonsMethodBranchless(domain, detectorDesc);

            THEN("A^t A x should be close to the original data")
            {
                auto AtAx = DataContainer(domain);

                op.apply(dataDomain, dataRange);
                op.applyAdjoint(dataRange, AtAx);

                auto cmp = RealVector_t(sizeDomain.prod());
                cmp << 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0;

                REQUIRE_UNARY(isCwiseApprox(DataContainer(domain, cmp), AtAx));
                REQUIRE_EQ(dataRange[0], Approx(5));
            }
        }

        WHEN("We have a single ray with 90 degrees")
        {
            VolumeData2D volDataCopy{volData};
            SinogramData2D sinoDataCopy{sinoData};
            geom.emplace_back(stc, ctr, Degree{90}, std::move(volDataCopy),
                              std::move(sinoDataCopy));

            auto detectorDesc = PlanarDetectorDescriptor(sizeRange, {geom});
            auto dataRange = DataContainer(detectorDesc);
            dataRange = 0;

            auto op = SiddonsMethodBranchless(domain, detectorDesc);

            THEN("A^t A x should be close to the original data")
            {
                auto AtAx = DataContainer(domain);

                op.apply(dataDomain, dataRange);
                op.applyAdjoint(dataRange, AtAx);

                auto cmp = RealVector_t(sizeDomain.prod());
                cmp << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

                REQUIRE_UNARY(isCwiseApprox(DataContainer(domain, cmp), AtAx));
                REQUIRE_EQ(dataRange[0], Approx(5));
            }
        }

        WHEN("We have a single ray with 270 degrees")
        {
            VolumeData2D volDataCopy{volData};
            SinogramData2D sinoDataCopy{sinoData};
            geom.emplace_back(stc, ctr, Degree{270}, std::move(volDataCopy),
                              std::move(sinoDataCopy));

            auto detectorDesc = PlanarDetectorDescriptor(sizeRange, {geom});
            auto dataRange = DataContainer(detectorDesc);
            dataRange = 0;

            auto op = SiddonsMethodBranchless(domain, detectorDesc);

            THEN("A^t A x should be close to the original data")
            {
                auto AtAx = DataContainer(domain);

                op.apply(dataDomain, dataRange);
                op.applyAdjoint(dataRange, AtAx);

                auto cmp = RealVector_t(sizeDomain.prod());
                cmp << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

                REQUIRE_UNARY(isCwiseApprox(DataContainer(domain, cmp), AtAx));
                REQUIRE_EQ(dataRange[0], Approx(5));
            }
        }

        // FIXME This does not yield the desired result/if fixed the overall results in a
        // reconstruction is bad
        /*
        WHEN("We have a single ray with 45 degrees")
        {
            geom.emplace_back(100, 5, 45 * pi_t / 180., domain, range);
            auto op = SiddonsMethod(domain, range, geom);

            THEN("A^t A x should be close to the original data")
            {
                auto AtAx = DataContainer(domain);

                op.apply(dataDomain, dataRange);
                op.applyAdjoint(dataRange, AtAx);

                std::cout << dataRange.getData().format(CommaInitFmt) << "\n";
                std::cout << AtAx.getData().format(CommaInitFmt) << "\n";

                auto cmp = RealVector_t(sizeDomain.prod());
                cmp << 10, 0, 0, 0, 0,
                        0, 10, 0, 0, 0,
                        0, 0, 10, 0, 0,
                        0, 0, 0, 10, 0,
                        0, 0, 0, 0, 10;

                REQUIRE_UNARY(cmp.isApprox(AtAx.getData(), 1e-2));
                REQUIRE_EQ(dataRange[0], Approx(7.071));
            }
        }

        WHEN("We have a single ray with 225 degrees")
        {
            geom.emplace_back(100, 5, 225 * pi_t / 180., domain, range);
            auto op = SiddonsMethod(domain, range, geom);

            THEN("A^t A x should be close to the original data")
            {
                auto AtAx = DataContainer(domain);

                op.apply(dataDomain, dataRange);
                op.applyAdjoint(dataRange, AtAx);

                std::cout << dataRange.getData().format(CommaInitFmt) << "\n";
                std::cout << AtAx.getData().format(CommaInitFmt) << "\n";

                auto cmp = RealVector_t(sizeDomain.prod());
                cmp << 10, 0, 0, 0, 0,
                        0, 10, 0, 0, 0,
                        0, 0, 10, 0, 0,
                        0, 0, 0, 10, 0,
                        0, 0, 0, 0, 10;

                REQUIRE_UNARY(cmp.isApprox(AtAx.getData()));
                REQUIRE_EQ(dataRange[0], Approx(7.071));
            }
        }*/

        // TODO fix this direction, currently it does not work correctly. Consider changing
        // geometry, to mirror stuff
        /*WHEN("We have a single ray with 135 degrees")
        {
            geom.emplace_back(100, 5, 135 * pi_t / 180., domain, range);
            auto op = SiddonsMethod(domain, range, geom);

            THEN("A^t A x should be close to the original data")
            {
                auto AtAx = DataContainer(domain);

                op.apply(dataDomain, dataRange);
                op.applyAdjoint(dataRange, AtAx);

                std::cout << dataRange.getData().format(CommaInitFmt) << "\n";
                std::cout << AtAx.getData().format(CommaInitFmt) << "\n";

                auto cmp = RealVector_t(sizeDomain.prod());
                cmp << 0, 0, 0, 0, 10,
                        0, 0, 0, 10, 0,
                        0, 0, 10, 0, 0,
                        0, 10, 0, 0, 0,
                        10, 0, 0, 0, 0;

                REQUIRE(cmp.isApprox(AtAx.getData()));
                REQUIRE_EQ(dataRange[0], Approx(7.071));
            }
        }*/
    }
}

TEST_CASE("SiddonMethod: Calls to functions of super class")
{
    // Turn logger of
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("A projector")
    {
        IndexVector_t volumeDims(2), sinoDims(2);
        const index_t volSize = 50;
        const index_t detectorSize = 50;
        const index_t numImgs = 50;
        volumeDims << volSize, volSize;
        sinoDims << detectorSize, numImgs;
        VolumeDescriptor volumeDescriptor(volumeDims);

        RealVector_t randomStuff(volumeDescriptor.getNumberOfCoefficients());
        randomStuff.setRandom();
        DataContainer volume(volumeDescriptor, randomStuff);

        auto stc = SourceToCenterOfRotation{20 * volSize};
        auto ctr = CenterOfRotationToDetector{volSize};

        std::vector<Geometry> geom;
        for (std::size_t i = 0; i < numImgs; i++) {
            real_t angle = static_cast<real_t>(i * 2) * pi_t / 50;
            geom.emplace_back(stc, ctr, Radian{angle}, VolumeData2D{Size2D{volumeDims}},
                              SinogramData2D{Size2D{sinoDims}});
        }
        PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
        DataContainer sino(sinoDescriptor);
        SiddonsMethodBranchless op(volumeDescriptor, sinoDescriptor);

        WHEN("Projector is cloned")
        {
            auto opClone = op.clone();
            auto sinoClone = DataContainer(sino.getDataDescriptor());
            auto volumeClone = DataContainer(volume.getDataDescriptor());

            THEN("Results do not change (may still be slightly different due to summation being "
                 "performed in a different order)")
            {
                op.apply(volume, sino);
                opClone->apply(volume, sinoClone);
                REQUIRE_UNARY(isApprox(sino, sinoClone));

                op.applyAdjoint(sino, volume);
                opClone->applyAdjoint(sino, volumeClone);
                REQUIRE_UNARY(isApprox(volume, volumeClone));
            }
        }
    }
}

TEST_CASE("SiddoneMethod: Output DataContainer is not zero initialized")
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

        SiddonsMethodBranchless op(volumeDescriptor, sinoDescriptor);

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
        DataContainer sino(sinoDescriptor);

        SiddonsMethodBranchless op(volumeDescriptor, sinoDescriptor);

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

TEST_CASE("SidddonMethod: Rays not intersecting the bounding box are present")
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
            VolumeData2D volDataCopy{volData};
            SinogramData2D sinoDataCopy{sinoData};
            geom.emplace_back(stc, ctr, Radian{0}, std::move(volDataCopy), std::move(sinoDataCopy),
                              PrincipalPointOffset{}, RotationOffset2D{-volSize, 0});

            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);
            sino = 1;

            SiddonsMethodBranchless op(volumeDescriptor, sinoDescriptor);

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
            VolumeData2D volDataCopy{volData};
            SinogramData2D sinoDataCopy{sinoData};
            geom.emplace_back(stc, ctr, Radian{0}, std::move(volDataCopy), std::move(sinoDataCopy),
                              PrincipalPointOffset{}, RotationOffset2D{volSize, 0});

            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);
            sino = 1;

            SiddonsMethodBranchless op(volumeDescriptor, sinoDescriptor);

            // SiddonsMethod op(volumeDescriptor, sinoDescriptor, geom);

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
            VolumeData2D volDataCopy{volData};
            SinogramData2D sinoDataCopy{sinoData};
            geom.emplace_back(stc, ctr, Radian{pi_t / 2}, std::move(volDataCopy),
                              std::move(sinoDataCopy), PrincipalPointOffset{},
                              RotationOffset2D{0, -volSize});

            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);
            sino = 1;

            SiddonsMethodBranchless op(volumeDescriptor, sinoDescriptor);

            // SiddonsMethod op(volumeDescriptor, sinoDescriptor, geom);

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
            VolumeData2D volDataCopy{volData};
            SinogramData2D sinoDataCopy{sinoData};
            geom.emplace_back(stc, ctr, Radian{pi_t / 2}, std::move(volDataCopy),
                              std::move(sinoDataCopy), PrincipalPointOffset{},
                              RotationOffset2D{0, volSize});

            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);
            sino = 1;

            SiddonsMethodBranchless op(volumeDescriptor, sinoDescriptor);

            // SiddonsMethod op(volumeDescriptor, sinoDescriptor, geom);

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
        auto volData = VolumeData3D{Size3D{volumeDims}};
        auto sinoData = SinogramData3D{Size3D{sinoDims}};

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
            WHEN("Tracing along axis-aligned ray through origins")
            {
                INFO("Tracing along a ", ali[i], "-axis-aligned ray with negative ", neg[i],
                     "-coodinate of origin");
                VolumeData3D volDataCopy{volData};
                SinogramData3D sinoDataCopy{sinoData};
                geom.emplace_back(stc, ctr, std::move(volDataCopy), std::move(sinoDataCopy),
                                  RotationAngles3D{Gamma{gamma[i]}, Beta{beta[i]}, Alpha{alpha[i]}},
                                  PrincipalPointOffset2D{0, 0},
                                  RotationOffset3D{-offsetx[i], -offsety[i], -offsetz[i]});

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer sino(sinoDescriptor);
                sino = 1;

                SiddonsMethodBranchless op(volumeDescriptor, sinoDescriptor);

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

TEST_CASE("SiddonMethod: Axis-aligned rays are present")
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
            WHEN("An axis-aligned ray with an angle of different angles passes through the center "
                 "of a pixel")
            {
                INFO("An axis-aligned ray with an angle of", angles[i],
                     " radians passes through the center of a pixel");
                VolumeData2D volDataCopy{volData};
                SinogramData2D sinoDataCopy{sinoData};
                geom.emplace_back(stc, ctr, Radian{angles[i]}, std::move(volDataCopy),
                                  std::move(sinoDataCopy));

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer sino(sinoDescriptor);

                SiddonsMethodBranchless op(volumeDescriptor, sinoDescriptor);
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

        WHEN("A y-axis-aligned ray runs along a voxel boundary")
        {
            VolumeData2D volDataCopy{volData};
            SinogramData2D sinoDataCopy{sinoData};
            geom.emplace_back(SourceToCenterOfRotation{volSize * 2000}, ctr, Radian{0},
                              std::move(volDataCopy), std::move(sinoDataCopy),
                              PrincipalPointOffset{0}, RotationOffset2D{-0.5, 0});
            // SiddonsMethod op(volumeDescriptor, sinoDescriptor, geom);
            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);

            SiddonsMethodBranchless op(volumeDescriptor, sinoDescriptor);
            THEN("The result of projecting through a pixel is the value of the pixel with the "
                 "higher index")
            {
                for (index_t j = 0; j < volSize; j++) {
                    volume = 0;
                    volume(volSize / 2, j) = 1;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0], Approx(1.0));
                }

                AND_THEN("The backprojection yields the exact adjoint")
                {
                    sino[0] = 1;
                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isApprox(volume, DataContainer(volumeDescriptor, backProj[0])));
                }
            }
        }

        WHEN("A y-axis-aligned ray runs along the right volume boundary")
        {
            // For Siddon's values in the range [0,boxMax) are considered, a ray running along
            // boxMax should be ignored
            VolumeData2D volDataCopy{volData};
            SinogramData2D sinoDataCopy{sinoData};
            geom.emplace_back(SourceToCenterOfRotation{volSize * 2000}, ctr, Radian{0},
                              std::move(volDataCopy), std::move(sinoDataCopy),
                              PrincipalPointOffset{0}, RotationOffset2D{volSize * 0.5, 0});
            // SiddonsMethod op(volumeDescriptor, sinoDescriptor, geom);
            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);

            SiddonsMethodBranchless op(volumeDescriptor, sinoDescriptor);

            THEN("The result of projecting is zero")
            {
                volume = 0;
                op.apply(volume, sino);
                REQUIRE_EQ(sino[0], Approx(0.0));

                AND_THEN("The result of backprojection is also zero")
                {
                    sino[0] = 1;

                    op.applyAdjoint(sino, volume);
                    DataContainer zero(volumeDescriptor);
                    zero = 0;
                    REQUIRE_UNARY(isApprox(volume, zero, epsilon));
                }
            }
        }

        /**
         * CPU version exhibits slightly different behaviour, ignoring all ray running along a
         * bounding box border
         */
        // backProj[0] <<  1, 0, 0, 0, 0,
        //             1, 0, 0, 0, 0,
        //             1, 0, 0, 0, 0,
        //             1, 0, 0, 0, 0,
        //             1, 0, 0, 0, 0;

        // WHEN("A y-axis-aligned ray runs along the left volume boundary") {
        //     geom.emplace_back(volSize*2000,volSize,0.0,volumeDescriptor,sinoDescriptor,0.0,volSize/2.0);
        //     SiddonsMethod op(volumeDescriptor,sinoDescriptor,geom);
        //     THEN("The result of projecting through a pixel is exactly the pixel's value") {
        //         for (index_t j=0; j<volSize;j++) {
        //             volume = 0;
        //             volume(0,j) = 1;

        //             op.apply(volume,sino);
        //             REQUIRE_EQ(sino[0], 1);
        //         }

        //         AND_THEN("The backprojection yields the exact adjoint") {
        //             sino[0] = 1;
        //             op.applyAdjoint(sino,volume);
        //             REQUIRE_UNARY(volume.getData().isApprox(backProj[0]));
        //         }
        //     }
        // }
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
                VolumeData3D volDataCopy{volData};
                SinogramData3D sinoDataCopy{sinoData};
                geom.emplace_back(stc, ctr, std::move(volDataCopy), std::move(sinoDataCopy),
                                  RotationAngles3D{Gamma{gamma[i]}, Beta{beta[i]}});

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer sino(sinoDescriptor);

                SiddonsMethodBranchless op(volumeDescriptor, sinoDescriptor);
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
                        REQUIRE_EQ(sino[0], Approx(1));
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

        real_t offsetx[numCases];
        real_t offsety[numCases];

        offsetx[0] = volSize / 2.0;
        offsetx[3] = -(volSize / 2.0);
        offsetx[1] = 0.0;
        offsetx[4] = 0.0;
        offsetx[5] = -(volSize / 2.0);
        offsetx[2] = (volSize / 2.0);

        offsety[0] = 0.0;
        offsety[3] = 0.0;
        offsety[1] = volSize / 2.0;
        offsety[4] = -(volSize / 2.0);
        offsety[5] = -(volSize / 2.0);
        offsety[2] = (volSize / 2.0);

        backProj[0] << 0, 0, 0, 1, 0, 0, 0, 0, 0,

            0, 0, 0, 1, 0, 0, 0, 0, 0,

            0, 0, 0, 1, 0, 0, 0, 0, 0;

        backProj[1] << 0, 1, 0, 0, 0, 0, 0, 0, 0,

            0, 1, 0, 0, 0, 0, 0, 0, 0,

            0, 1, 0, 0, 0, 0, 0, 0, 0;

        backProj[2] << 1, 0, 0, 0, 0, 0, 0, 0, 0,

            1, 0, 0, 0, 0, 0, 0, 0, 0,

            1, 0, 0, 0, 0, 0, 0, 0, 0;

        al[0] = "left border";
        al[1] = "bottom border";
        al[2] = "bottom left edge";
        al[3] = "right border";
        al[4] = "top border";
        al[5] = "top right edge";

        for (index_t i = 0; i < numCases / 2; i++) {
            WHEN("A z-axis-aligned ray runs along the corners and edges of the volume")
            {
                INFO("A z-axis-aligned ray runs along the ", al[i], " of the volume");
                // x-ray source must be very far from the volume center to make testing of the op
                // backprojection simpler
                VolumeData3D volDataCopy{volData};
                SinogramData3D sinoDataCopy{sinoData};
                geom.emplace_back(SourceToCenterOfRotation{volSize * 2000}, ctr,
                                  std::move(volDataCopy), std::move(sinoDataCopy),
                                  RotationAngles3D{Gamma{0}}, PrincipalPointOffset2D{0, 0},
                                  RotationOffset3D{-offsetx[i], -offsety[i], 0});
                // SiddonsMethod op(volumeDescriptor, sinoDescriptor, geom);
                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer sino(sinoDescriptor);

                SiddonsMethodBranchless op(volumeDescriptor, sinoDescriptor);
                THEN("The result of projecting through a voxel is exactly the voxel's value")
                {
                    for (index_t j = 0; j < volSize; j++) {
                        volume = 0;
                        switch (i) {
                            case 0:
                                volume(0, volSize / 2, j) = 1;
                                break;
                            case 1:
                                volume(volSize / 2, 0, j) = 1;
                                break;
                                break;
                            case 2:
                                volume(0, 0, j) = 1;
                                break;
                            default:
                                break;
                        }

                        op.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx(1));
                    }

                    AND_THEN("The backprojection yields the exact adjoints")
                    {
                        sino[0] = 1;
                        op.applyAdjoint(sino, volume);

                        REQUIRE_UNARY(
                            isApprox(volume, DataContainer(volumeDescriptor, backProj[i])));
                    }
                }
            }
        }

        for (index_t i = 3; i < numCases; i++) {
            WHEN("A z-axis-aligned ray runs along the corners and edges of the volume")
            {
                INFO("A z-axis-aligned ray runs along the ", al[i], " of the volume");
                // x-ray source must be very far from the volume center to make testing of the op
                // backprojection simpler
                VolumeData3D volDataCopy{volData};
                SinogramData3D sinoDataCopy{sinoData};
                geom.emplace_back(SourceToCenterOfRotation{volSize * 2000}, ctr,
                                  std::move(volDataCopy), std::move(sinoDataCopy),
                                  RotationAngles3D{Gamma{0}}, PrincipalPointOffset2D{0, 0},
                                  RotationOffset3D{-offsetx[i], -offsety[i], 0});
                // SiddonsMethod op(volumeDescriptor, sinoDescriptor, geom);
                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer sino(sinoDescriptor);

                SiddonsMethodBranchless op(volumeDescriptor, sinoDescriptor);
                THEN("The result of projecting is zero")
                {
                    volume = 1;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0], Approx(0));

                    AND_THEN("The result of backprojection is also zero")
                    {
                        sino[0] = 1;
                        op.applyAdjoint(sino, volume);

                        DataContainer zero(volumeDescriptor);
                        zero = 0;
                        REQUIRE_UNARY(isApprox(volume, zero, epsilon));
                    }
                }
            }
        }
    }

    GIVEN("A 2D setting with multiple projection angles")
    {
        IndexVector_t volumeDims(2), sinoDims(2);
        const index_t volSize = 5;
        const index_t detectorSize = 1;
        const index_t numImgs = 4;
        volumeDims << volSize, volSize;
        sinoDims << detectorSize, numImgs;
        VolumeDescriptor volumeDescriptor(volumeDims);
        DataContainer volume(volumeDescriptor);

        auto stc = SourceToCenterOfRotation{20 * volSize};
        auto ctr = CenterOfRotationToDetector{volSize};

        std::vector<Geometry> geom;

        WHEN("Both x- and y-axis-aligned rays are present")
        {
            geom.emplace_back(stc, ctr, Degree{0}, VolumeData2D{Size2D{volumeDims}},
                              SinogramData2D{Size2D{sinoDims}});
            geom.emplace_back(stc, ctr, Degree{90}, VolumeData2D{Size2D{volumeDims}},
                              SinogramData2D{Size2D{sinoDims}});
            geom.emplace_back(stc, ctr, Degree{180}, VolumeData2D{Size2D{volumeDims}},
                              SinogramData2D{Size2D{sinoDims}});
            geom.emplace_back(stc, ctr, Degree{270}, VolumeData2D{Size2D{volumeDims}},
                              SinogramData2D{Size2D{sinoDims}});

            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);

            SiddonsMethodBranchless op(volumeDescriptor, sinoDescriptor);

            THEN("Values are accumulated correctly along each ray's path")
            {
                volume = 0;

                // set only values along the rays' path to one to make sure interpolation is dones
                // correctly
                for (index_t i = 0; i < volSize; i++) {
                    volume(i, volSize / 2) = 1;
                    volume(volSize / 2, i) = 1;
                }

                op.apply(volume, sino);
                for (index_t i = 0; i < numImgs; i++)
                    REQUIRE_EQ(sino[i], Approx(5.0));

                AND_THEN("Backprojection yields the exact adjoint")
                {
                    RealVector_t cmp(volSize * volSize);

                    cmp << 0, 0, 10, 0, 0, 0, 0, 10, 0, 0, 10, 10, 20, 10, 10, 0, 0, 10, 0, 0, 0, 0,
                        10, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isApprox(volume, DataContainer(volumeDescriptor, cmp)));
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

            SiddonsMethodBranchless op(volumeDescriptor, sinoDescriptor);

            THEN("Values are accumulated correctly along each ray's path")
            {
                volume = 0;

                // set only values along the rays' path to one to make sure interpolation is dones
                // correctly
                for (index_t i = 0; i < volSize; i++) {
                    volume(i, volSize / 2, volSize / 2) = 1;
                    volume(volSize / 2, i, volSize / 2) = 1;
                    volume(volSize / 2, volSize / 2, i) = 1;
                }

                op.apply(volume, sino);
                for (index_t i = 0; i < numImgs; i++)
                    REQUIRE_EQ(sino[i], Approx(3.0));

                AND_THEN("Backprojection yields the exact adjoint")
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
}

TEST_CASE("SiddonsMethod: Projection under an angle")
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
            // direction
            VolumeData2D volDataCopy{volData};
            SinogramData2D sinoDataCopy{sinoData};
            geom.emplace_back(stc, ctr, Radian{-pi_t / 6}, std::move(volDataCopy),
                              std::move(sinoDataCopy));

            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);

            SiddonsMethodBranchless op(volumeDescriptor, sinoDescriptor);

            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                volume(3, 0) = 0;
                volume(2, 0) = 0;
                volume(2, 1) = 0;

                volume(1, 2) = 0;
                volume(1, 3) = 0;
                volume(0, 3) = 0;
                // volume(2,2 also hit due to numerical errors)
                volume(2, 2) = 0;

                op.apply(volume, sino);
                DataContainer zero(sinoDescriptor);
                zero = 0;
                CHECK(std::abs(sino[0]) <= Approx(0.0001f).epsilon(epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    volume(3, 0) = 1;
                    volume(2, 0) = 2;
                    volume(2, 1) = 3;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0], Approx(2 * std::sqrt(3.f) + 2));

                    // on the other side of the center
                    volume = 0;
                    volume(1, 2) = 3;
                    volume(1, 3) = 2;
                    volume(0, 3) = 1;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0], Approx(2 * std::sqrt(3.f) + 2));

                    sino[0] = 1;

                    RealVector_t expected(volSize * volSize);
                    expected << 0, 0, 2 - 2 / std::sqrt(3.f), 4 / std::sqrt(3.f) - 2, 0, 0,
                        2 / std::sqrt(3.f), 0, 0, 2 / std::sqrt(3.f), 0, 0, 4 / std::sqrt(3.f) - 2,
                        2 - 2 / std::sqrt(3.f), 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isApprox(volume, DataContainer(volumeDescriptor, expected)));
                }
            }
        }

        WHEN("Projecting under an angle of 30 degrees and ray enters through the right border")
        {
            // In this case the ray exits through a border along the main ray direction, but enters
            // through a border not along the main direction
            VolumeData2D volDataCopy{volData};
            SinogramData2D sinoDataCopy{sinoData};
            geom.emplace_back(stc, ctr, Radian{-pi_t / 6}, std::move(volDataCopy),
                              std::move(sinoDataCopy), PrincipalPointOffset{0},
                              RotationOffset2D{std::sqrt(3.f), 0});
            // SiddonsMethod op(volumeDescriptor, sinoDescriptor, geom);
            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);

            SiddonsMethodBranchless op(volumeDescriptor, sinoDescriptor);

            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                volume(3, 1) = 0;
                volume(3, 2) = 0;
                volume(3, 3) = 0;
                volume(2, 3) = 0;

                op.apply(volume, sino);
                DataContainer zero(sinoDescriptor);
                zero = 0;
                REQUIRE_UNARY(isApprox(sino, zero, epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    volume(3, 1) = 4;
                    volume(3, 2) = 3;
                    volume(3, 3) = 2;
                    volume(2, 3) = 1;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0], Approx(14 - 4 * std::sqrt(3.f)));

                    sino[0] = 1;

                    RealVector_t expected(volSize * volSize);
                    expected << 0, 0, 0, 0, 0, 0, 0, 4 - 2 * std::sqrt(3.f), 0, 0, 0,
                        2 / std::sqrt(3.f), 0, 0, 2 - 2 / std::sqrt(3.f), 4 / std::sqrt(3.f) - 2;

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(
                        isApprox(volume, DataContainer(volumeDescriptor, expected), epsilon));
                }
            }
        }

        WHEN("Projecting under an angle of 30 degrees and ray exits through the left border")
        {
            // In this case the ray enters through a border along the main ray direction, but exits
            // through a border not along the main direction
            VolumeData2D volDataCopy{volData};
            SinogramData2D sinoDataCopy{sinoData};
            geom.emplace_back(stc, ctr, Radian{-pi_t / 6}, std::move(volDataCopy),
                              std::move(sinoDataCopy), PrincipalPointOffset{0},
                              RotationOffset2D{-std::sqrt(3.f), 0});
            // SiddonsMethod op(volumeDescriptor, sinoDescriptor, geom);
            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);

            SiddonsMethodBranchless op(volumeDescriptor, sinoDescriptor);

            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                volume(0, 0) = 0;
                volume(1, 0) = 0;
                volume(0, 1) = 0;
                volume(0, 2) = 0;

                op.apply(volume, sino);
                DataContainer zero(sinoDescriptor);
                zero = 0;
                REQUIRE_UNARY(isApprox(sino, zero, epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    volume(1, 0) = 1;
                    volume(0, 0) = 2;
                    volume(0, 1) = 3;
                    volume(0, 2) = 4;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0], Approx(14 - 4 * std::sqrt(3.f)));

                    sino[0] = 1;

                    RealVector_t expected(volSize * volSize);
                    expected << 4 / std::sqrt(3.f) - 2, 2 - 2 / std::sqrt(3.f), 0, 0,
                        2 / std::sqrt(3.f), 0, 0, 0, 4 - 2 * std::sqrt(3.f), 0, 0, 0, 0, 0, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isApprox(volume, DataContainer(volumeDescriptor, expected)));
                }
            }
        }

        WHEN("Projecting under an angle of 30 degrees and ray only intersects a single pixel")
        {
            VolumeData2D volDataCopy{volData};
            SinogramData2D sinoDataCopy{sinoData};
            geom.emplace_back(stc, ctr, Radian{-pi_t / 6}, std::move(volDataCopy),
                              std::move(sinoDataCopy), PrincipalPointOffset{0},
                              RotationOffset2D{-2 - std::sqrt(3.f) / 2, 0});
            // SiddonsMethod op(volumeDescriptor, sinoDescriptor, geom);
            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);

            SiddonsMethodBranchless op(volumeDescriptor, sinoDescriptor);

            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                volume(0, 0) = 0;

                op.apply(volume, sino);
                DataContainer zero(sinoDescriptor);
                zero = 0;
                REQUIRE_UNARY(isApprox(sino, zero, epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    volume(0, 0) = 1;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0], Approx(1 / std::sqrt(3.f)));

                    sino[0] = 1;

                    RealVector_t expected(volSize * volSize);
                    expected << 1 / std::sqrt(3.f), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(
                        isApprox(volume, DataContainer(volumeDescriptor, expected), epsilon));
                }
            }
        }

        WHEN("Projecting under an angle of 120 degrees and ray goes through center of volume")
        {
            // In this case the ray enters and exits the volume through the borders along the main
            // direction
            VolumeData2D volDataCopy{volData};
            SinogramData2D sinoDataCopy{sinoData};
            geom.emplace_back(stc, ctr, Radian{-2 * pi_t / 3}, std::move(volDataCopy),
                              std::move(sinoDataCopy));
            // SiddonsMethod op(volumeDescriptor, sinoDescriptor, geom);
            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);

            SiddonsMethodBranchless op(volumeDescriptor, sinoDescriptor);

            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                volume(0, 0) = 0;
                volume(0, 1) = 0;
                volume(1, 1) = 0;
                volume(2, 2) = 0;
                volume(3, 2) = 0;
                volume(3, 3) = 0;
                // volume(1,2) hit due to numerical error
                volume(1, 2) = 0;

                op.apply(volume, sino);
                DataContainer zero(sinoDescriptor);
                zero = 0;
                CHECK(std::abs(sino[0]) <= Approx(0.0001f).epsilon(epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    volume(0, 0) = 1;
                    volume(0, 1) = 2;
                    volume(1, 1) = 3;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0], Approx(2 * std::sqrt(3.f) + 2));

                    // on the other side of the center
                    volume = 0;
                    volume(2, 2) = 3;
                    volume(3, 2) = 2;
                    volume(3, 3) = 1;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0], Approx(2 * std::sqrt(3.f) + 2));

                    sino[0] = 1;

                    RealVector_t expected(volSize * volSize);

                    expected << 4 / std::sqrt(3.f) - 2, 0, 0, 0, 2 - 2 / std::sqrt(3.f),
                        2 / std::sqrt(3.f), 0, 0, 0, 0, 2 / std::sqrt(3.f), 2 - 2 / std::sqrt(3.f),
                        0, 0, 0, 4 / std::sqrt(3.f) - 2;

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isApprox(volume, DataContainer(volumeDescriptor, expected)));
                }
            }
        }

        WHEN("Projecting under an angle of 120 degrees and ray enters through the top border")
        {
            // In this case the ray exits through a border along the main ray direction, but enters
            // through a border not along the main direction
            VolumeData2D volDataCopy{volData};
            SinogramData2D sinoDataCopy{sinoData};
            geom.emplace_back(stc, ctr, Radian{-2 * pi_t / 3}, std::move(volDataCopy),
                              std::move(sinoDataCopy), PrincipalPointOffset{0},
                              RotationOffset2D{0, std::sqrt(3.f)});
            // SiddonsMethod op(volumeDescriptor, sinoDescriptor, geom);
            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);

            SiddonsMethodBranchless op(volumeDescriptor, sinoDescriptor);

            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                volume(0, 2) = 0;
                volume(0, 3) = 0;
                volume(1, 3) = 0;
                volume(2, 3) = 0;

                op.apply(volume, sino);
                DataContainer zero(sinoDescriptor);
                zero = 0;
                REQUIRE_UNARY(isApprox(sino, zero, epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    volume(0, 2) = 1;
                    volume(0, 3) = 2;
                    volume(1, 3) = 3;
                    volume(2, 3) = 4;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0], Approx(14 - 4 * std::sqrt(3.f)));

                    sino[0] = 1;

                    RealVector_t expected(volSize * volSize);

                    expected << 0, 0, 0, 0, 0, 0, 0, 0, 2 - 2 / std::sqrt(3.f), 0, 0, 0,
                        4 / std::sqrt(3.f) - 2, 2 / std::sqrt(3.f), 4 - 2 * std::sqrt(3.f), 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(
                        isApprox(volume, DataContainer(volumeDescriptor, expected), epsilon));
                }
            }
        }

        WHEN("Projecting under an angle of 120 degrees and ray exits through the bottom border")
        {
            // In this case the ray enters through a border along the main ray direction, but exits
            // through a border not along the main direction
            VolumeData2D volDataCopy{volData};
            SinogramData2D sinoDataCopy{sinoData};
            geom.emplace_back(stc, ctr, Radian{-2 * pi_t / 3}, std::move(volDataCopy),
                              std::move(sinoDataCopy), PrincipalPointOffset{0},
                              RotationOffset2D{0, -std::sqrt(3.f)});
            // SiddonsMethod op(volumeDescriptor, sinoDescriptor, geom);
            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);

            SiddonsMethodBranchless op(volumeDescriptor, sinoDescriptor);

            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                volume(1, 0) = 0;
                volume(2, 0) = 0;
                volume(3, 0) = 0;
                volume(3, 1) = 0;

                op.apply(volume, sino);
                DataContainer zero(sinoDescriptor);
                zero = 0;
                REQUIRE_UNARY(isApprox(sino, zero, epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    volume(1, 0) = 4;
                    volume(2, 0) = 3;
                    volume(3, 0) = 2;
                    volume(3, 1) = 1;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0], Approx(14 - 4 * std::sqrt(3.f)));

                    sino[0] = 1;

                    RealVector_t expected(volSize * volSize);

                    expected << 0, 4 - 2 * std::sqrt(3.f), 2 / std::sqrt(3.f),
                        4 / std::sqrt(3.f) - 2, 0, 0, 0, 2 - 2 / std::sqrt(3.f), 0, 0, 0, 0, 0, 0,
                        0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isApprox(volume, DataContainer(volumeDescriptor, expected)));
                }
            }
        }

        WHEN("Projecting under an angle of 120 degrees and ray only intersects a single pixel")
        {
            // This is a special case that is handled separately in both forward and backprojection
            VolumeData2D volDataCopy{volData};
            SinogramData2D sinoDataCopy{sinoData};
            geom.emplace_back(stc, ctr, Radian{-2 * pi_t / 3}, std::move(volDataCopy),
                              std::move(sinoDataCopy), PrincipalPointOffset{0},
                              RotationOffset2D{0, -2 - std::sqrt(3.f) / 2});
            // SiddonsMethod op(volumeDescriptor, sinoDescriptor, geom);
            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);

            SiddonsMethodBranchless op(volumeDescriptor, sinoDescriptor);

            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                volume(3, 0) = 0;

                op.apply(volume, sino);
                DataContainer zero(sinoDescriptor);
                zero = 0;
                REQUIRE_UNARY(isApprox(sino, zero, epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    volume(3, 0) = 1;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0], Approx(1 / std::sqrt(3.f)).epsilon(epsilon));

                    sino[0] = 1;

                    RealVector_t expected(volSize * volSize);
                    expected << 0, 0, 0, 1 / std::sqrt(3.f), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(
                        isApprox(volume, DataContainer(volumeDescriptor, expected), epsilon));
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
            VolumeData3D volDataCopy{volData};
            SinogramData3D sinoDataCopy{sinoData};
            geom.emplace_back(stc, ctr, std::move(volDataCopy), std::move(sinoDataCopy),
                              RotationAngles3D{Gamma{pi_t / 6}});

            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);

            SiddonsMethodBranchless op(volumeDescriptor, sinoDescriptor);

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
                    REQUIRE_EQ(sino[0], Approx(3 * std::sqrt(3.f) - 1));

                    sino[0] = 1;
                    backProj << 0, 0, 0, 0, 1 - 1 / std::sqrt(3.f), std::sqrt(3.f) - 1, 0, 0, 0,

                        0, 0, 0, 0, 2 / std::sqrt(3.f), 0, 0, 0, 0,

                        0, 0, 0, std::sqrt(3.f) - 1, 1 - 1 / std::sqrt(3.f), 0, 0, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(
                        isApprox(volume, DataContainer(volumeDescriptor, backProj), epsilon));
                }
            }
        }

        WHEN("A ray with an angle of 30 degrees enters through the right border")
        {
            // In this case the ray enters through a border orthogonal to a non-main direction
            VolumeData3D volDataCopy{volData};
            SinogramData3D sinoDataCopy{sinoData};
            geom.emplace_back(stc, ctr, std::move(volDataCopy), std::move(sinoDataCopy),
                              RotationAngles3D{Gamma{pi_t / 6}}, PrincipalPointOffset2D{0, 0},
                              RotationOffset3D{1, 0, 0});
            // SiddonsMethod op(volumeDescriptor, sinoDescriptor, geom);
            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);

            SiddonsMethodBranchless op(volumeDescriptor, sinoDescriptor);

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
                    REQUIRE_EQ(sino[0], Approx(1 - 2 / std::sqrt(3.f) + 3 * std::sqrt(3.f)));

                    sino[0] = 1;
                    backProj << 0, 0, 0, 0, 0, 1 - 1 / std::sqrt(3.f), 0, 0, 0,

                        0, 0, 0, 0, 0, 2 / std::sqrt(3.f), 0, 0, 0,

                        0, 0, 0, 0, std::sqrt(3.f) - 1, 1 - 1 / std::sqrt(3.f), 0, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isApprox(volume, DataContainer(volumeDescriptor, backProj)));
                }
            }
        }

        WHEN("A ray with an angle of 30 degrees exits through the left border")
        {
            // In this case the ray exit through a border orthogonal to a non-main direction
            VolumeData3D volDataCopy{volData};
            SinogramData3D sinoDataCopy{sinoData};
            geom.emplace_back(stc, ctr, std::move(volDataCopy), std::move(sinoDataCopy),
                              RotationAngles3D{Gamma{pi_t / 6}}, PrincipalPointOffset2D{0, 0},
                              RotationOffset3D{-1, 0, 0});
            // SiddonsMethod op(volumeDescriptor, sinoDescriptor, geom);
            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);

            SiddonsMethodBranchless op(volumeDescriptor, sinoDescriptor);

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
                    REQUIRE_EQ(
                        sino[0],
                        Approx(3 * std::sqrt(3.f) + 1 - 2 / std::sqrt(3.f)).epsilon(epsilon));

                    sino[0] = 1;
                    backProj << 0, 0, 0, 1 - 1 / std::sqrt(3.f), std::sqrt(3.f) - 1, 0, 0, 0, 0,

                        0, 0, 0, 2 / std::sqrt(3.f), 0, 0, 0, 0, 0,

                        0, 0, 0, 1 - 1 / std::sqrt(3.f), 0, 0, 0, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(
                        isApprox(volume, DataContainer(volumeDescriptor, backProj), epsilon));
                }
            }
        }

        WHEN("A ray with an angle of 30 degrees only intersects a single voxel")
        {
            // special case - no interior voxels, entry and exit voxels are the same
            VolumeData3D volDataCopy{volData};
            SinogramData3D sinoDataCopy{sinoData};
            geom.emplace_back(stc, ctr, std::move(volDataCopy), std::move(sinoDataCopy),
                              RotationAngles3D{Gamma{pi_t / 6}}, PrincipalPointOffset2D{0, 0},
                              RotationOffset3D{-2, 0, 0});
            // SiddonsMethod op(volumeDescriptor, sinoDescriptor, geom);
            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer sino(sinoDescriptor);

            SiddonsMethodBranchless op(volumeDescriptor, sinoDescriptor);

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
}
