/**
 * @file test_BinaryMethod.cpp
 *
 * @brief Test for BinaryMethod class
 *
 * @author David Frank - initial code
 * @author Nikola Dinev - rewrite and major extensions
 * @author Tobias Lasser - minor fixes
 */

#include "doctest/doctest.h"
#include "BinaryMethod.h"
#include "Logger.h"
#include "testHelpers.h"
#include "VolumeDescriptor.h"
#include "PlanarDetectorDescriptor.h"

using namespace elsa;
using namespace elsa::geometry;
using namespace doctest;

// TODO: remove this and replace with checkApproxEq
using doctest::Approx;

TEST_CASE("BinaryMethod: Testing with only one ray")
{
    // Turn logger of
    Logger::setLevel(Logger::LogLevel::OFF);

    IndexVector_t sizeDomain(2);
    sizeDomain << 5, 5;

    IndexVector_t sizeRange(2);
    sizeRange << 1, 1;

    auto domain = VolumeDescriptor(sizeDomain);
    // auto range = VolumeDescriptor(sizeRange);

    auto stc = SourceToCenterOfRotation{100};
    auto ctr = CenterOfRotationToDetector{5};
    auto volData = VolumeData2D{Size2D{sizeDomain}};
    auto sinoData = SinogramData2D{Size2D{sizeRange}};

    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "",
                                 " << ", ";");

    GIVEN("A BinaryMethod for 2D and a domain data with all ones")
    {
        std::vector<Geometry> geom;

        auto dataDomain = DataContainer(domain);
        dataDomain = 1;

        WHEN("We have a single ray with 0 degrees")
        {
            VolumeData2D volDataCopy{volData};
            SinogramData2D sinoDataCopy{sinoData};
            geom.emplace_back(stc, ctr, Radian{0}, std::move(volDataCopy), std::move(sinoDataCopy));

            auto range = PlanarDetectorDescriptor(sizeRange, geom);
            auto op = BinaryMethod(domain, range);

            auto dataRange = DataContainer(range);
            dataRange = 0;

            THEN("A^t A x should be close to the original data")
            {
                auto AtAx = DataContainer(domain);

                op.apply(dataDomain, dataRange);

                REQUIRE_EQ(dataRange[0], Approx(5));

                op.applyAdjoint(dataRange, AtAx);

                auto cmp = RealVector_t(sizeDomain.prod());
                cmp << 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0;
                DataContainer tmpCmp(domain, cmp);

                REQUIRE_UNARY(isCwiseApprox(tmpCmp, AtAx));
            }
        }

        WHEN("We have a single ray with 180 degrees")
        {
            VolumeData2D volDataCopy{volData};
            SinogramData2D sinoDataCopy{sinoData};
            geom.emplace_back(stc, ctr, Degree{180}, std::move(volDataCopy),
                              std::move(sinoDataCopy));

            // auto op = BinaryMethod(domain, range, geom);
            auto range = PlanarDetectorDescriptor(sizeRange, geom);
            auto op = BinaryMethod(domain, range);
            auto dataRange = DataContainer(range);
            dataRange = 0;

            THEN("A^t A x should be close to the original data")
            {
                auto AtAx = DataContainer(domain);

                op.apply(dataDomain, dataRange);

                REQUIRE_EQ(dataRange[0], Approx(5));

                op.applyAdjoint(dataRange, AtAx);

                auto cmp = RealVector_t(sizeDomain.prod());
                cmp << 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0;
                DataContainer tmpCmp(domain, cmp);

                REQUIRE_UNARY(isCwiseApprox(tmpCmp, AtAx));
            }
        }

        WHEN("We have a single ray with 90 degrees")
        {
            VolumeData2D volDataCopy{volData};
            SinogramData2D sinoDataCopy{sinoData};
            geom.emplace_back(stc, ctr, Degree{90}, std::move(volDataCopy),
                              std::move(sinoDataCopy));

            // auto op = BinaryMethod(domain, range, geom);
            auto range = PlanarDetectorDescriptor(sizeRange, geom);
            auto op = BinaryMethod(domain, range);
            auto dataRange = DataContainer(range);
            dataRange = 0;

            THEN("A^t A x should be close to the original data")
            {
                auto AtAx = DataContainer(domain);

                op.apply(dataDomain, dataRange);

                REQUIRE_EQ(dataRange[0], Approx(5));

                op.applyAdjoint(dataRange, AtAx);

                auto cmp = RealVector_t(sizeDomain.prod());
                cmp << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                DataContainer tmpCmp(domain, cmp);

                REQUIRE_UNARY(isCwiseApprox(tmpCmp, AtAx));
            }
        }

        WHEN("We have a single ray with 270 degrees")
        {
            VolumeData2D volDataCopy{volData};
            SinogramData2D sinoDataCopy{sinoData};
            geom.emplace_back(stc, ctr, Degree{270}, std::move(volDataCopy),
                              std::move(sinoDataCopy));

            // auto op = BinaryMethod(domain, range, geom);
            auto range = PlanarDetectorDescriptor(sizeRange, geom);
            auto op = BinaryMethod(domain, range);
            auto dataRange = DataContainer(range);
            dataRange = 0;

            THEN("A^t A x should be close to the original data")
            {
                auto AtAx = DataContainer(domain);

                op.apply(dataDomain, dataRange);

                REQUIRE_EQ(dataRange[0], Approx(5));

                op.applyAdjoint(dataRange, AtAx);

                auto cmp = RealVector_t(sizeDomain.prod());
                cmp << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                DataContainer tmpCmp(domain, cmp);

                REQUIRE_UNARY(isCwiseApprox(tmpCmp, AtAx));
            }
        }

        WHEN("We have a single ray with 45 degrees")
        {
            VolumeData2D volDataCopy{volData};
            SinogramData2D sinoDataCopy{sinoData};
            geom.emplace_back(stc, ctr, Degree{45}, std::move(volDataCopy),
                              std::move(sinoDataCopy));

            // auto op = BinaryMethod(domain, range, geom);
            auto range = PlanarDetectorDescriptor(sizeRange, geom);
            auto op = BinaryMethod(domain, range);
            auto dataRange = DataContainer(range);
            dataRange = 0;

            THEN("A^t A x should be close to the original data")
            {
                auto AtAx = DataContainer(domain);

                op.apply(dataDomain, dataRange);
                op.applyAdjoint(dataRange, AtAx);

                auto cmp = RealVector_t(sizeDomain.prod());
                cmp << 9, 0, 0, 0, 0, 9, 9, 0, 0, 0, 0, 9, 9, 0, 0, 0, 0, 9, 9, 0, 0, 0, 0, 9, 9;
                DataContainer tmpCmp(domain, cmp);

                REQUIRE_UNARY(isCwiseApprox(tmpCmp, AtAx));
            }
        }

        WHEN("We have a single ray with 225 degrees")
        {
            VolumeData2D volDataCopy{volData};
            SinogramData2D sinoDataCopy{sinoData};
            geom.emplace_back(stc, ctr, Degree{225}, std::move(volDataCopy),
                              std::move(sinoDataCopy));

            // auto op = BinaryMethod(domain, range, geom);
            auto range = PlanarDetectorDescriptor(sizeRange, geom);
            auto op = BinaryMethod(domain, range);

            auto dataRange = DataContainer(range);
            dataRange = 0;

            // This test case is a little awkward, but the Problem is inside of Geometry, with tiny
            // rounding erros this will not give exactly a ray with direction of (1/1), rather
            // (1.000001/0.99999), then the traversal is rather correct again, but in the operator
            // in bigger settings, this error will not be a problem.
            THEN("A^t A x should be close to the original data")
            {
                auto AtAx = DataContainer(domain);

                op.apply(dataDomain, dataRange);
                op.applyAdjoint(dataRange, AtAx);

                // std::cout << AtAx.getData().format(CommaInitFmt) << "\n\n";

                auto cmp = RealVector_t(sizeDomain.prod());
                cmp << 9, 9, 0, 0, 0, 0, 9, 9, 0, 0, 0, 0, 9, 9, 0, 0, 0, 0, 9, 9, 0, 0, 0, 0, 9;
                DataContainer tmpCmp(domain, cmp);

                REQUIRE_UNARY(isCwiseApprox(tmpCmp, AtAx));
            }
        }
    }
}

TEST_CASE("BinaryMethod: Testing with only 1 rays for 4 angles")
{
    // Turn logger of
    Logger::setLevel(Logger::LogLevel::OFF);

    IndexVector_t sizeDomain(2);
    sizeDomain << 5, 5;

    IndexVector_t sizeRange(2);
    sizeRange << 1, 4;

    auto domain = VolumeDescriptor(sizeDomain);
    // auto range = VolumeDescriptor(sizeRange);

    auto stc = SourceToCenterOfRotation{100};
    auto ctr = CenterOfRotationToDetector{5};
    auto volData = VolumeData2D{Size2D{sizeDomain}};
    auto sinoData = SinogramData2D{Size2D{sizeRange}};

    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "",
                                 " << ", ";");

    GIVEN("A BinaryMethod for 2D and a domain data with all ones")
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

            // auto op = BinaryMethod(domain, range, geom);
            auto range = PlanarDetectorDescriptor(sizeRange, geom);
            auto op = BinaryMethod(domain, range);

            auto dataRange = DataContainer(range);
            dataRange = 0;

            THEN("A^t A x should be close to the original data")
            {
                auto AtAx = DataContainer(domain);

                op.apply(dataDomain, dataRange);
                op.applyAdjoint(dataRange, AtAx);

                // std::cout << AtAx.getData().format(CommaInitFmt) << "\n";

                auto cmp = RealVector_t(sizeDomain.prod());
                cmp << 0, 0, 10, 0, 0, 0, 0, 10, 0, 0, 10, 10, 20, 10, 10, 0, 0, 10, 0, 0, 0, 0, 10,
                    0, 0;
                DataContainer tmpCmp(domain, cmp);

                REQUIRE_UNARY(isCwiseApprox(tmpCmp, AtAx));
            }

            WHEN("A clone is created from the projector")
            {
                auto cloneOp = op.clone();

                THEN("The results will stay the same")
                {
                    auto AtAx = DataContainer(domain);

                    cloneOp->apply(dataDomain, dataRange);
                    cloneOp->applyAdjoint(dataRange, AtAx);

                    auto cmp = RealVector_t(sizeDomain.prod());
                    cmp << 0, 0, 10, 0, 0, 0, 0, 10, 0, 0, 10, 10, 20, 10, 10, 0, 0, 10, 0, 0, 0, 0,
                        10, 0, 0;
                    DataContainer tmpCmp(domain, cmp);

                    REQUIRE_UNARY(isCwiseApprox(tmpCmp, AtAx));
                }
            }
        }
    }
}

TEST_CASE("BinaryMethod: Testing different setup")
{
    // Turn logger of
    Logger::setLevel(Logger::LogLevel::OFF);

    IndexVector_t sizeDomain(2);
    sizeDomain << 5, 5;

    IndexVector_t sizeRange(2);
    sizeRange << 5, 1;

    auto domain = VolumeDescriptor(sizeDomain);
    // auto range = VolumeDescriptor(sizeRange);

    auto stc = SourceToCenterOfRotation{100};
    auto ctr = CenterOfRotationToDetector{5};
    auto volData = VolumeData2D{Size2D{sizeDomain}};
    auto sinoData = SinogramData2D{Size2D{sizeRange}};

    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "",
                                 " << ", ";");

    GIVEN("A BinaryMethod with 1 angle at 0 degree")
    {
        std::vector<Geometry> geom;
        VolumeData2D volDataCopy{volData};
        SinogramData2D sinoDataCopy{sinoData};
        geom.emplace_back(stc, ctr, Degree{0}, std::move(volDataCopy), std::move(sinoDataCopy));

        // auto op = BinaryMethod(domain, range, geom);
        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BinaryMethod(domain, range);

        //        THEN("It is not spd")
        //        {
        //            REQUIRE_FALSE(op.isSpd());
        //        }

        THEN("Domain descriptor is still the same")
        {
            auto& retDescriptor = op.getDomainDescriptor();

            CHECK_EQ(retDescriptor.getNumberOfCoefficientsPerDimension()(0), 5);
            CHECK_EQ(retDescriptor.getNumberOfCoefficientsPerDimension()(1), 5);
        }

        THEN("Domain descriptor is still the same")
        {
            auto& retDescriptor = op.getRangeDescriptor();

            CHECK_EQ(retDescriptor.getNumberOfCoefficientsPerDimension()(0), 5);
            CHECK_EQ(retDescriptor.getNumberOfCoefficientsPerDimension()(1), 1);
        }

        WHEN("We have domain data with only ones")
        {
            auto dataDomain = DataContainer(domain);
            dataDomain = 1;

            auto dataRange = DataContainer(range);
            dataRange = 0;

            THEN("A^t A x should be close to the original data")
            {
                auto AtAx = DataContainer(domain);

                op.apply(dataDomain, dataRange);

                RealVector_t res = RealVector_t::Constant(sizeRange.prod(), 1, 5);
                DataContainer tmpRes(range, res);
                REQUIRE_UNARY(isCwiseApprox(tmpRes, dataRange));

                op.applyAdjoint(dataRange, AtAx);

                auto cmp = RealVector_t(sizeDomain.prod());
                cmp << 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5;
                DataContainer tmpCmp(domain, cmp);

                REQUIRE_UNARY(isCwiseApprox(tmpCmp, AtAx));
            }
        }
    }

    GIVEN("A traversal with 5 rays at 180 degrees")
    {
        std::vector<Geometry> geom;
        VolumeData2D volDataCopy{volData};
        SinogramData2D sinoDataCopy{sinoData};
        geom.emplace_back(stc, ctr, Degree{180}, std::move(volDataCopy), std::move(sinoDataCopy));

        // auto op = BinaryMethod(domain, range, geom);
        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BinaryMethod(domain, range);

        THEN("A^t A x should be close to the original data")
        {
            auto dataDomain = DataContainer(domain);
            dataDomain = 1;

            auto dataRange = DataContainer(range);
            dataRange = 0;

            auto AtAx = DataContainer(domain);

            op.apply(dataDomain, dataRange);

            RealVector_t res = RealVector_t::Constant(sizeRange.prod(), 1, 5);
            DataContainer tmpRes(range, res);
            REQUIRE_UNARY(isCwiseApprox(tmpRes, dataRange));

            op.applyAdjoint(dataRange, AtAx);

            auto cmp = RealVector_t(sizeDomain.prod());
            cmp << 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5;
            DataContainer tmpCmp(domain, cmp);

            REQUIRE_UNARY(isCwiseApprox(tmpCmp, AtAx));
        }
    }

    GIVEN("A traversal with 5 rays at 90 degrees")
    {
        std::vector<Geometry> geom;
        VolumeData2D volDataCopy{volData};
        SinogramData2D sinoDataCopy{sinoData};
        geom.emplace_back(stc, ctr, Degree{90}, std::move(volDataCopy), std::move(sinoDataCopy));

        // auto op = BinaryMethod(domain, range, geom);
        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BinaryMethod(domain, range);

        THEN("A^t A x should be close to the original data")
        {
            auto dataDomain = DataContainer(domain);
            dataDomain = 1;

            auto dataRange = DataContainer(range);
            dataRange = 0;

            auto AtAx = DataContainer(domain);

            op.apply(dataDomain, dataRange);

            RealVector_t res = RealVector_t::Constant(sizeRange.prod(), 1, 5);
            DataContainer tmpRes(range, res);
            REQUIRE_UNARY(isApprox(tmpRes, dataRange));

            op.applyAdjoint(dataRange, AtAx);

            auto cmp = RealVector_t(sizeDomain.prod());
            cmp << 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5;
            DataContainer tmpCmp(domain, cmp);

            REQUIRE_UNARY(isCwiseApprox(tmpCmp, AtAx));
        }
    }

    GIVEN("A traversal with 5 rays at 270 degrees")
    {
        std::vector<Geometry> geom;
        VolumeData2D volDataCopy{volData};
        SinogramData2D sinoDataCopy{sinoData};
        geom.emplace_back(stc, ctr, Degree{270}, std::move(volDataCopy), std::move(sinoDataCopy));

        // auto op = BinaryMethod(domain, range, geom);
        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BinaryMethod(domain, range);

        THEN("A^t A x should be close to the original data")
        {
            auto dataDomain = DataContainer(domain);
            dataDomain = 1;

            auto dataRange = DataContainer(range);
            dataRange = 0;

            auto AtAx = DataContainer(domain);

            op.apply(dataDomain, dataRange);

            RealVector_t res = RealVector_t::Constant(sizeRange.prod(), 1, 5);
            DataContainer tmpRes(range, res);
            REQUIRE_UNARY(isApprox(tmpRes, dataRange));

            op.applyAdjoint(dataRange, AtAx);

            auto cmp = RealVector_t(sizeDomain.prod());
            cmp << 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5;
            DataContainer tmpCmp(domain, cmp);

            REQUIRE_UNARY(isCwiseApprox(tmpCmp, AtAx));
        }
    }
}

TEST_CASE("BinaryMethod: Calls to functions of super class")
{
    // Turn logger of
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("A projector")
    {
        IndexVector_t volumeDims(2), sizeRange(2);
        const index_t volSize = 50;
        const index_t detectorSize = 50;
        const index_t numImgs = 50;
        volumeDims << volSize, volSize;
        sizeRange << detectorSize, numImgs;
        VolumeDescriptor volumeDescriptor(volumeDims);
        VolumeDescriptor sinoDescriptor(sizeRange);

        RealVector_t randomStuff(volumeDescriptor.getNumberOfCoefficients());
        randomStuff.setRandom();
        DataContainer volume(volumeDescriptor, randomStuff);
        DataContainer sino(sinoDescriptor);

        auto stc = SourceToCenterOfRotation{20 * volSize};
        auto ctr = CenterOfRotationToDetector{volSize};

        std::vector<Geometry> geom;
        for (index_t i = 0; i < numImgs; i++) {
            auto angle = static_cast<real_t>(i * 2) * pi_t / 50;
            geom.emplace_back(stc, ctr, Radian{angle}, VolumeData2D{Size2D{volumeDims}},
                              SinogramData2D{Size2D{sizeRange}});
        }

        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BinaryMethod(volumeDescriptor, range);

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

TEST_CASE("BinaryMethod: Output DataContainer is not zero initialized")
{
    // Turn logger of
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("A 2D setting")
    {
        IndexVector_t volumeDims(2), sizeRange(2);
        const index_t volSize = 5;
        const index_t detectorSize = 1;
        const index_t numImgs = 1;
        volumeDims << volSize, volSize;
        sizeRange << detectorSize, numImgs;
        VolumeDescriptor volumeDescriptor(volumeDims);
        VolumeDescriptor sinoDescriptor(sizeRange);

        DataContainer volume(volumeDescriptor);
        DataContainer sino(sinoDescriptor);

        auto stc = SourceToCenterOfRotation{20 * volSize};
        auto ctr = CenterOfRotationToDetector{volSize};
        auto volData = VolumeData2D{Size2D{volumeDims}};
        auto sinoData = SinogramData2D{Size2D{sizeRange}};

        std::vector<Geometry> geom;
        geom.emplace_back(stc, ctr, Radian{0}, std::move(volData), std::move(sinoData));

        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BinaryMethod(volumeDescriptor, range);

        WHEN("Sinogram conatainer is not zero initialized and we project through an empty volume")
        {
            volume = 0;
            sino = 1;

            THEN("Result is zero")
            {
                op.apply(volume, sino);

                DataContainer zero(sinoDescriptor);
                zero = 0;
                REQUIRE_UNARY(isCwiseApprox(sino, zero));
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
                REQUIRE_UNARY(isCwiseApprox(volume, zero));
            }
        }
    }

    GIVEN("A 3D setting")
    {
        IndexVector_t volumeDims(3), sizeRange(3);
        const index_t volSize = 3;
        const index_t detectorSize = 1;
        const index_t numImgs = 1;
        volumeDims << volSize, volSize, volSize;
        sizeRange << detectorSize, detectorSize, numImgs;
        VolumeDescriptor volumeDescriptor(volumeDims);
        VolumeDescriptor sinoDescriptor(sizeRange);

        auto stc = SourceToCenterOfRotation{20 * volSize};
        auto ctr = CenterOfRotationToDetector{volSize};
        auto volData = VolumeData3D{Size3D{volumeDims}};
        auto sinoData = SinogramData3D{Size3D{sizeRange}};

        DataContainer volume(volumeDescriptor);
        DataContainer sino(sinoDescriptor);

        std::vector<Geometry> geom;
        geom.emplace_back(stc, ctr, std::move(volData), std::move(sinoData),
                          RotationAngles3D{Gamma{0}});

        // BinaryMethod op(volumeDescriptor, sinoDescriptor, geom);
        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BinaryMethod(volumeDescriptor, range);

        WHEN("Sinogram conatainer is not zero initialized and we project through an empty volume")
        {
            volume = 0;
            sino = 1;

            THEN("Result is zero")
            {
                op.apply(volume, sino);

                DataContainer zero(sinoDescriptor);
                zero = 0;
                REQUIRE_UNARY(isCwiseApprox(sino, zero));
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
                REQUIRE_UNARY(isCwiseApprox(volume, zero));
            }
        }
    }
}

TEST_CASE("BinaryMethod: Rays not intersecting the bounding box are present")
{
    // Turn logger of
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("A 2D setting")
    {
        IndexVector_t volumeDims(2), sizeRange(2);
        const index_t volSize = 5;
        const index_t detectorSize = 1;
        const index_t numImgs = 1;
        volumeDims << volSize, volSize;
        sizeRange << detectorSize, numImgs;
        VolumeDescriptor volumeDescriptor(volumeDims);
        VolumeDescriptor sinoDescriptor(sizeRange);

        auto stc = SourceToCenterOfRotation{20 * volSize};
        auto ctr = CenterOfRotationToDetector{volSize};
        auto volData = VolumeData2D{Size2D{volumeDims}};
        auto sinoData = SinogramData2D{Size2D{sizeRange}};

        DataContainer volume(volumeDescriptor);
        DataContainer sino(sinoDescriptor);
        volume = 1;
        sino = 0;

        std::vector<Geometry> geom;

        WHEN("Tracing along a y-axis-aligned ray with a negative x-coordinate of origin")
        {
            VolumeData2D volDataCopy{volData};
            SinogramData2D sinoDataCopy{sinoData};
            geom.emplace_back(stc, ctr, Radian{0}, std::move(volDataCopy), std::move(sinoDataCopy),
                              PrincipalPointOffset{}, RotationOffset2D{-volSize, 0});

            // BinaryMethod op(volumeDescriptor, sinoDescriptor, geom);
            auto range = PlanarDetectorDescriptor(sizeRange, geom);
            auto op = BinaryMethod(volumeDescriptor, range);

            THEN("Result of forward projection is zero")
            {
                op.apply(volume, sino);

                DataContainer zeroSino(sinoDescriptor);
                zeroSino = 0;
                REQUIRE_UNARY(isCwiseApprox(sino, zeroSino));

                AND_THEN("Result of backprojection is zero")
                {
                    op.applyAdjoint(sino, volume);

                    DataContainer zeroVolume(volumeDescriptor);
                    zeroVolume = 0;
                    REQUIRE_UNARY(isCwiseApprox(volume, zeroVolume));
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

            // BinaryMethod op(volumeDescriptor, sinoDescriptor, geom);
            auto range = PlanarDetectorDescriptor(sizeRange, geom);
            auto op = BinaryMethod(volumeDescriptor, range);

            THEN("Result of forward projection is zero")
            {
                op.apply(volume, sino);

                DataContainer zeroSino(sinoDescriptor);
                zeroSino = 0;
                REQUIRE_UNARY(isCwiseApprox(sino, zeroSino));

                AND_THEN("Result of backprojection is zero")
                {
                    op.applyAdjoint(sino, volume);

                    DataContainer zeroVolume(volumeDescriptor);
                    zeroVolume = 0;
                    REQUIRE_UNARY(isCwiseApprox(volume, zeroVolume));
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

            // BinaryMethod op(volumeDescriptor, sinoDescriptor, geom);
            auto range = PlanarDetectorDescriptor(sizeRange, geom);
            auto op = BinaryMethod(volumeDescriptor, range);

            THEN("Result of forward projection is zero")
            {
                op.apply(volume, sino);

                DataContainer zeroSino(sinoDescriptor);
                zeroSino = 0;
                REQUIRE_UNARY(isCwiseApprox(sino, zeroSino));

                AND_THEN("Result of backprojection is zero")
                {
                    op.applyAdjoint(sino, volume);

                    DataContainer zeroVolume(volumeDescriptor);
                    zeroVolume = 0;
                    REQUIRE_UNARY(isCwiseApprox(volume, zeroVolume));
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

            // BinaryMethod op(volumeDescriptor, sinoDescriptor, geom);
            auto range = PlanarDetectorDescriptor(sizeRange, geom);
            auto op = BinaryMethod(volumeDescriptor, range);

            THEN("Result of forward projection is zero")
            {
                op.apply(volume, sino);

                DataContainer zeroSino(sinoDescriptor);
                zeroSino = 0;
                REQUIRE_UNARY(isCwiseApprox(sino, zeroSino));

                AND_THEN("Result of backprojection is zero")
                {
                    op.applyAdjoint(sino, volume);

                    DataContainer zeroVolume(volumeDescriptor);
                    zeroVolume = 0;
                    REQUIRE_UNARY(isCwiseApprox(volume, zeroVolume));
                }
            }
        }
    }

    GIVEN("A 3D setting")
    {
        IndexVector_t volumeDims(3), sizeRange(3);
        const index_t volSize = 5;
        const index_t detectorSize = 1;
        const index_t numImgs = 1;
        volumeDims << volSize, volSize, volSize;
        sizeRange << detectorSize, detectorSize, numImgs;
        VolumeDescriptor volumeDescriptor(volumeDims);
        VolumeDescriptor sinoDescriptor(sizeRange);

        DataContainer volume(volumeDescriptor);
        DataContainer sino(sinoDescriptor);
        volume = 1;
        sino = 1;

        auto stc = SourceToCenterOfRotation{20 * volSize};
        auto ctr = CenterOfRotationToDetector{volSize};
        auto volData = VolumeData3D{Size3D{volumeDims}};
        auto sinoData = SinogramData3D{Size3D{sizeRange}};

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

        // TODO: fix tests for i < numCases
        for (int i = 0; i < 1; i++) {
            WHEN("Tracing rays along different axis")
            {
                INFO("Tracing along a ", ali[i], "-axis-aligned ray with negative ", neg[i],
                     "-coodinate of origin");
                VolumeData3D volDataCopy{volData};
                SinogramData3D sinoDataCopy{sinoData};
                geom.emplace_back(stc, ctr, std::move(volDataCopy), std::move(sinoDataCopy),
                                  RotationAngles3D{Gamma{gamma[i]}, Beta{beta[i]}, Alpha{alpha[i]}},
                                  PrincipalPointOffset2D{0, 0},
                                  RotationOffset3D{-offsetx[i], -offsety[i], -offsetz[i]});

                // BinaryMethod op(volumeDescriptor, sinoDescriptor, geom);
                auto range = PlanarDetectorDescriptor(sizeRange, geom);
                auto op = BinaryMethod(volumeDescriptor, range);

                THEN("Result of forward projection is zero")
                {
                    op.apply(volume, sino);

                    DataContainer zeroSino(sinoDescriptor);
                    zeroSino = 0;
                    REQUIRE_UNARY(isCwiseApprox(sino, zeroSino));

                    AND_THEN("Result of backprojection is zero")
                    {
                        op.applyAdjoint(sino, volume);

                        DataContainer zeroVolume(volumeDescriptor);
                        zeroVolume = 0;
                        REQUIRE_UNARY(isCwiseApprox(volume, zeroVolume));
                    }
                }
            }
        }
    }
}

TEST_CASE("BinaryMethod: Axis-aligned rays are present")
{
    // Turn logger of
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("A 2D setting with a single ray")
    {
        IndexVector_t volumeDims(2), sizeRange(2);
        const index_t volSize = 5;
        const index_t detectorSize = 1;
        const index_t numImgs = 1;
        volumeDims << volSize, volSize;
        sizeRange << detectorSize, numImgs;
        VolumeDescriptor volumeDescriptor(volumeDims);
        VolumeDescriptor sinoDescriptor(sizeRange);

        DataContainer volume(volumeDescriptor);
        DataContainer sino(sinoDescriptor);

        auto stc = SourceToCenterOfRotation{20 * volSize};
        auto ctr = CenterOfRotationToDetector{volSize};
        auto volData = VolumeData2D{Size2D{volumeDims}};
        auto sinoData = SinogramData2D{Size2D{sizeRange}};

        std::vector<Geometry> geom;

        const index_t numCases = 4;
        const real_t angles[numCases] = {0.0, pi_t / 2, pi_t, 3 * pi_t / 2};
        RealVector_t backProj[2];
        backProj[0].resize(volSize * volSize);
        backProj[1].resize(volSize * volSize);
        backProj[1] << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

        backProj[0] << 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0;

        // TODO: fix tests for i < numCases
        for (index_t i = 0; i < 1; i++) {
            WHEN("Axis-aligned ray through the center of the pixel")
            {
                INFO("An axis-aligned ray with an angle of ", angles[i],
                     " radians passes through the center of a pixel");
                VolumeData2D volDataCopy{volData};
                SinogramData2D sinoDataCopy{sinoData};
                geom.emplace_back(stc, ctr, Radian{angles[i]}, std::move(volDataCopy),
                                  std::move(sinoDataCopy));
                // BinaryMethod op(volumeDescriptor, sinoDescriptor, geom);
                auto range = PlanarDetectorDescriptor(sizeRange, geom);
                auto op = BinaryMethod(volumeDescriptor, range);
                THEN("The result of projecting through a pixel is exactly the pixel value")
                {
                    for (index_t j = 0; j < volSize; j++) {
                        volume = 0;
                        if (i % 2 == 0) {
                            IndexVector_t coord(2);
                            coord << volSize / 2, j;
                            volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 1;
                        } else {
                            IndexVector_t coord(2);
                            coord << j, volSize / 2;
                            volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 1;
                        }

                        op.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx(1));
                    }

                    AND_THEN("The backprojection sets the values of all hit pixels to the detector "
                             "value")
                    {
                        op.applyAdjoint(sino, volume);

                        DataContainer res(volumeDescriptor, backProj[i % 2]);
                        REQUIRE_UNARY(isCwiseApprox(volume, res));
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

            // BinaryMethod op(volumeDescriptor, sinoDescriptor, geom);
            auto range = PlanarDetectorDescriptor(sizeRange, geom);
            auto op = BinaryMethod(volumeDescriptor, range);
            THEN("The result of projecting through a pixel is the value of the pixel with the "
                 "higher index")
            {
                for (index_t j = 0; j < volSize; j++) {
                    volume = 0;
                    IndexVector_t coord(2);
                    coord << volSize / 2, j;
                    volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 1;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0], Approx(1.0));
                }

                AND_THEN("The backprojection yields the exact adjoint")
                {
                    sino[0] = 1;
                    op.applyAdjoint(sino, volume);

                    DataContainer res(volumeDescriptor, backProj[0]);
                    REQUIRE_UNARY(isCwiseApprox(volume, res));
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

            // BinaryMethod op(volumeDescriptor, sinoDescriptor, geom);
            auto range = PlanarDetectorDescriptor(sizeRange, geom);
            auto op = BinaryMethod(volumeDescriptor, range);

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
                    REQUIRE_UNARY(isCwiseApprox(volume, zero));
                }
            }
        }

        backProj[0] << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0;

        WHEN("A y-axis-aligned ray runs along the left volume boundary")
        {
            VolumeData2D volDataCopy{volData};
            SinogramData2D sinoDataCopy{sinoData};
            geom.emplace_back(SourceToCenterOfRotation{volSize * 2000}, ctr, Radian{0},
                              std::move(volDataCopy), std::move(sinoDataCopy),
                              PrincipalPointOffset{0}, RotationOffset2D{-volSize / 2.0, 0});

            // BinaryMethod op(volumeDescriptor, sinoDescriptor, geom);
            auto range = PlanarDetectorDescriptor(sizeRange, geom);
            auto op = BinaryMethod(volumeDescriptor, range);
            THEN("The result of projecting through a pixel is exactly the pixel's value")
            {
                for (index_t j = 0; j < volSize; j++) {
                    volume = 0;
                    volume(0, j) = 1;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0], Approx(1));
                }

                AND_THEN("The backprojection yields the exact adjoint")
                {
                    sino[0] = 1;
                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(
                        isCwiseApprox(volume, DataContainer(volumeDescriptor, backProj[0])));
                }
            }
        }
    }

    GIVEN("A 3D setting with a single ray")
    {
        IndexVector_t volumeDims(3), sizeRange(3);
        const index_t volSize = 3;
        const index_t detectorSize = 1;
        const index_t numImgs = 1;
        volumeDims << volSize, volSize, volSize;
        sizeRange << detectorSize, detectorSize, numImgs;
        VolumeDescriptor volumeDescriptor(volumeDims);
        VolumeDescriptor sinoDescriptor(sizeRange);

        DataContainer volume(volumeDescriptor);
        DataContainer sino(sinoDescriptor);

        auto stc = SourceToCenterOfRotation{20 * volSize};
        auto ctr = CenterOfRotationToDetector{volSize};
        auto volData = VolumeData3D{Size3D{volumeDims}};
        auto sinoData = SinogramData3D{Size3D{sizeRange}};

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

        // TODO: fix tests for i < numCases
        for (index_t i = 0; i < 1; i++) {
            WHEN("Tracing an axis-aligned ray trough the pixel center")
            {
                INFO("A ", al[i], "-axis-aligned ray passes through the center of a pixel");
                VolumeData3D volDataCopy{volData};
                SinogramData3D sinoDataCopy{sinoData};
                geom.emplace_back(stc, ctr, std::move(volDataCopy), std::move(sinoDataCopy),
                                  RotationAngles3D{Gamma{gamma[i]}, Beta{beta[i]}});

                // BinaryMethod op(volumeDescriptor, sinoDescriptor, geom);
                auto range = PlanarDetectorDescriptor(sizeRange, geom);
                auto op = BinaryMethod(volumeDescriptor, range);
                THEN("The result of projecting through a voxel is exactly the voxel value")
                {
                    for (index_t j = 0; j < volSize; j++) {
                        volume = 0;
                        if (i / 2 == 0) {
                            IndexVector_t coord(3);
                            coord << volSize / 2, volSize / 2, j;
                            volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 1;
                        } else if (i / 2 == 1) {
                            IndexVector_t coord(3);
                            coord << j, volSize / 2, volSize / 2;
                            volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 1;
                        } else if (i / 2 == 2) {
                            IndexVector_t coord(3);
                            coord << volSize / 2, j, volSize / 2;
                            volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 1;
                        }

                        op.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx(1));
                    }

                    AND_THEN("The backprojection sets the values of all hit voxels to the detector "
                             "value")
                    {
                        op.applyAdjoint(sino, volume);

                        DataContainer res(volumeDescriptor, backProj[i / 2]);
                        REQUIRE_UNARY(isCwiseApprox(volume, res));
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
        al[1] = "right border";
        al[2] = "top border";
        al[3] = "bottom border";
        al[4] = "top right edge";
        al[5] = "bottom left edge";

        // TODO: fix tests for i < numCases / 2
        for (index_t i = 0; i < 1; i++) {
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

                // BinaryMethod op(volumeDescriptor, sinoDescriptor, geom);
                auto range = PlanarDetectorDescriptor(sizeRange, geom);
                auto op = BinaryMethod(volumeDescriptor, range);
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
                            isCwiseApprox(volume, DataContainer(volumeDescriptor, backProj[i])));
                    }
                }
            }
        }

        // TODO: fix tests for i < numCases
        for (index_t i = numCases / 2; i < 1; i++) {
            WHEN("A z-axis-aligned ray runs along the edges and corners of the volume")
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

                // BinaryMethod op(volumeDescriptor, sinoDescriptor, geom);
                auto range = PlanarDetectorDescriptor(sizeRange, geom);
                auto op = BinaryMethod(volumeDescriptor, range);
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
                        REQUIRE_UNARY(isCwiseApprox(volume, zero));
                    }
                }
            }
        }
    }

    GIVEN("A 2D setting with multiple projection angles")
    {
        IndexVector_t volumeDims(2), sizeRange(2);
        const index_t volSize = 5;
        const index_t detectorSize = 1;
        const index_t numImgs = 4;
        volumeDims << volSize, volSize;
        sizeRange << detectorSize, numImgs;
        VolumeDescriptor volumeDescriptor(volumeDims);
        VolumeDescriptor sinoDescriptor(sizeRange);
        DataContainer volume(volumeDescriptor);
        DataContainer sino(sinoDescriptor);

        auto stc = SourceToCenterOfRotation{20 * volSize};
        auto ctr = CenterOfRotationToDetector{volSize};

        std::vector<Geometry> geom;

        WHEN("Both x- and y-axis-aligned rays are present")
        {
            geom.emplace_back(stc, ctr, Degree{0}, VolumeData2D{Size2D{volumeDims}},
                              SinogramData2D{Size2D{sizeRange}});
            geom.emplace_back(stc, ctr, Degree{90}, VolumeData2D{Size2D{volumeDims}},
                              SinogramData2D{Size2D{sizeRange}});
            geom.emplace_back(stc, ctr, Degree{180}, VolumeData2D{Size2D{volumeDims}},
                              SinogramData2D{Size2D{sizeRange}});
            geom.emplace_back(stc, ctr, Degree{270}, VolumeData2D{Size2D{volumeDims}},
                              SinogramData2D{Size2D{sizeRange}});

            // BinaryMethod op(volumeDescriptor, sinoDescriptor, geom);
            auto range = PlanarDetectorDescriptor(sizeRange, geom);
            auto op = BinaryMethod(volumeDescriptor, range);

            THEN("Values are accumulated correctly along each ray's path")
            {
                volume = 0;

                // set only values along the rays' path to one to make sure interpolation is dones
                // correctly
                for (index_t i = 0; i < volSize; i++) {
                    IndexVector_t coord(2);
                    coord << i, volSize / 2;
                    volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 1;
                    coord << volSize / 2, i;
                    volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 1;
                }

                op.apply(volume, sino);
                for (index_t i = 0; i < numImgs; i++)
                    REQUIRE_EQ(sino[i], Approx(5.0));

                AND_THEN("Backprojection yields the exact adjoint")
                {
                    RealVector_t cmp(volSize * volSize);

                    cmp << 0, 0, 10, 0, 0, 0, 0, 10, 0, 0, 10, 10, 20, 10, 10, 0, 0, 10, 0, 0, 0, 0,
                        10, 0, 0;
                    DataContainer tmpCmp(volumeDescriptor, cmp);

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isCwiseApprox(volume, tmpCmp));
                }
            }
        }
    }

    GIVEN("A 3D setting with multiple projection angles")
    {
        IndexVector_t volumeDims(3), sizeRange(3);
        const index_t volSize = 3;
        const index_t detectorSize = 1;
        const index_t numImgs = 6;
        volumeDims << volSize, volSize, volSize;
        sizeRange << detectorSize, detectorSize, numImgs;
        VolumeDescriptor volumeDescriptor(volumeDims);
        VolumeDescriptor sinoDescriptor(sizeRange);
        DataContainer volume(volumeDescriptor);
        DataContainer sino(sinoDescriptor);

        auto stc = SourceToCenterOfRotation{20 * volSize};
        auto ctr = CenterOfRotationToDetector{volSize};

        std::vector<Geometry> geom;

        WHEN("x-, y and z-axis-aligned rays are present")
        {
            real_t beta[numImgs] = {0.0, 0.0, 0.0, 0.0, pi_t / 2, 3 * pi_t / 2};
            real_t gamma[numImgs] = {0.0, pi_t, pi_t / 2, 3 * pi_t / 2, pi_t / 2, 3 * pi_t / 2};

            for (index_t i = 0; i < numImgs; i++)
                geom.emplace_back(stc, ctr, VolumeData3D{Size3D{volumeDims}},
                                  SinogramData3D{Size3D{sizeRange}},
                                  RotationAngles3D{Gamma{gamma[i]}, Beta{beta[i]}});

            // BinaryMethod op(volumeDescriptor, sinoDescriptor, geom);
            auto range = PlanarDetectorDescriptor(sizeRange, geom);
            auto op = BinaryMethod(volumeDescriptor, range);

            THEN("Values are accumulated correctly along each ray's path")
            {
                volume = 0;

                // set only values along the rays' path to one to make sure interpolation is dones
                // correctly
                for (index_t i = 0; i < volSize; i++) {
                    IndexVector_t coord(3);
                    coord << i, volSize / 2, volSize / 2;
                    volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 1;
                    coord << volSize / 2, i, volSize / 2;
                    volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 1;
                    coord << volSize / 2, volSize / 2, i;
                    volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 1;
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
                    DataContainer tmpCmp(volumeDescriptor, cmp);

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isCwiseApprox(volume, tmpCmp));
                }
            }
        }
    }
}

TEST_CASE("BinaryMethod: Projection under an angle")
{
    // Turn logger of
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("A 2D setting with a single ray")
    {
        IndexVector_t volumeDims(2), sizeRange(2);
        const index_t volSize = 4;
        const index_t detectorSize = 1;
        const index_t numImgs = 1;
        volumeDims << volSize, volSize;
        sizeRange << detectorSize, numImgs;
        VolumeDescriptor volumeDescriptor(volumeDims);
        VolumeDescriptor sinoDescriptor(sizeRange);

        DataContainer volume(volumeDescriptor);
        DataContainer sino(sinoDescriptor);

        auto stc = SourceToCenterOfRotation{20 * volSize};
        auto ctr = CenterOfRotationToDetector{volSize};
        auto volData = VolumeData2D{Size2D{volumeDims}};
        auto sinoData = SinogramData2D{Size2D{sizeRange}};

        std::vector<Geometry> geom;

        WHEN("Projecting under an angle of 30 degrees and ray goes through center of volume")
        {
            // In this case the ray enters and exits the volume through the borders along the main
            // direction
            VolumeData2D volDataCopy{volData};
            SinogramData2D sinoDataCopy{sinoData};
            geom.emplace_back(stc, ctr, Radian{-pi_t / 6}, std::move(volDataCopy),
                              std::move(sinoDataCopy));
            // BinaryMethod op(volumeDescriptor, sinoDescriptor, geom);
            auto range = PlanarDetectorDescriptor(sizeRange, geom);
            auto op = BinaryMethod(volumeDescriptor, range);

            THEN("Ray intersects the correct pixels")
            {
                //  Our volume: with left top being (0, 0) and bottom right (3,3)
                // 0, 0, 1, 1,
                // 0, 1, 1, 0,
                // 0, 1, 0, 0,
                // 1, 1, 0, 0

                volume = 1;
                volume(3, 0) = 0;
                volume(2, 0) = 0;
                volume(2, 1) = 0;
                volume(1, 1) = 0;
                volume(1, 2) = 0;
                volume(1, 3) = 0;
                volume(0, 3) = 0;

                op.apply(volume, sino);

                DataContainer sZero(sinoDescriptor);
                sZero = 0;
                CHECK_LE(std::abs(sino[0]), Approx(0.0001f).epsilon(epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    volume(3, 0) = 1;
                    volume(2, 0) = 2;
                    volume(2, 1) = 3;

                    op.apply(volume, sino);
                    CHECK_EQ(sino[0], Approx(6));

                    // on the other side of the center
                    volume = 0;
                    volume(1, 2) = 3;
                    volume(1, 3) = 2;
                    volume(0, 3) = 1;

                    op.apply(volume, sino);
                    CHECK_EQ(sino[0], Approx(6));

                    sino[0] = 1;

                    RealVector_t expected(volSize * volSize);
                    expected << 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0;

                    op.applyAdjoint(sino, volume);

                    DataContainer resVolume(volumeDescriptor, expected);
                    REQUIRE_UNARY(isCwiseApprox(volume, resVolume));
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
            // BinaryMethod op(volumeDescriptor, sinoDescriptor, geom);
            auto range = PlanarDetectorDescriptor(sizeRange, geom);
            auto op = BinaryMethod(volumeDescriptor, range);

            THEN("Ray intersects the correct pixels")
            {
                volume = 0;
                IndexVector_t coord(2);
                coord << 3, 1;
                volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 0;
                coord << 3, 2;
                volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 0;
                coord << 3, 3;
                volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 0;
                coord << 2, 3;
                volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 0;

                op.apply(volume, sino);
                DataContainer sZero(sinoDescriptor);
                sZero = 0;
                CHECK_UNARY(isApprox(sino, sZero, epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    coord << 3, 1;
                    volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 4;
                    coord << 3, 2;
                    volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 3;
                    coord << 3, 3;
                    volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 2;
                    coord << 2, 3;
                    volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 1;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0], Approx(10));

                    sino[0] = 1;

                    RealVector_t expected(volSize * volSize);
                    expected << 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1;
                    DataContainer resVolume(volumeDescriptor, expected);

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isCwiseApprox(volume, resVolume));
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
            // BinaryMethod op(volumeDescriptor, sinoDescriptor, geom);
            auto range = PlanarDetectorDescriptor(sizeRange, geom);
            auto op = BinaryMethod(volumeDescriptor, range);

            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                IndexVector_t coord(2);
                coord << 0, 0;
                volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 0;
                coord << 1, 0;
                volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 0;
                coord << 0, 1;
                volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 0;
                coord << 0, 2;
                volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 0;

                op.apply(volume, sino);
                DataContainer sZero(sinoDescriptor);
                sZero = 0;
                CHECK_UNARY(isApprox(sino, sZero, epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    coord << 1, 0;
                    volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 1;
                    coord << 0, 0;
                    volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 2;
                    coord << 0, 1;
                    volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 3;
                    coord << 0, 2;
                    volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 4;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0], Approx(10));

                    sino[0] = 1;

                    RealVector_t expected(volSize * volSize);
                    expected << 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0;
                    DataContainer resVolume(volumeDescriptor, expected);

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isCwiseApprox(volume, resVolume));
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
            // BinaryMethod op(volumeDescriptor, sinoDescriptor, geom);
            auto range = PlanarDetectorDescriptor(sizeRange, geom);
            auto op = BinaryMethod(volumeDescriptor, range);

            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                IndexVector_t coord(2);
                coord << 0, 0;
                volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 0;

                op.apply(volume, sino);
                DataContainer sZero(sinoDescriptor);
                sZero = 0;
                CHECK_UNARY(isApprox(sino, sZero, epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    coord << 0, 0;
                    volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 1;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0], Approx(1));

                    sino[0] = 1;

                    RealVector_t expected(volSize * volSize);
                    expected << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                    DataContainer resVolume(volumeDescriptor, expected);

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isCwiseApprox(volume, resVolume));
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
            // BinaryMethod op(volumeDescriptor, sinoDescriptor, geom);
            auto range = PlanarDetectorDescriptor(sizeRange, geom);
            auto op = BinaryMethod(volumeDescriptor, range);

            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                volume(0, 0) = 0;
                volume(0, 1) = 0;
                volume(1, 1) = 0;
                volume(2, 1) = 0;
                volume(2, 2) = 0;
                volume(3, 2) = 0;
                volume(3, 3) = 0;

                op.apply(volume, sino);
                DataContainer sZero(sinoDescriptor);
                sZero = 0;
                CHECK_UNARY(isApprox(sino, sZero, epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    volume = 0;
                    volume(0, 0) = 3;
                    volume(0, 1) = 2;
                    volume(1, 1) = 1;

                    op.apply(volume, sino);
                    CHECK_EQ(sino[0], Approx(6));

                    // on the other side of the center
                    volume = 0;
                    volume(2, 2) = 3;
                    volume(3, 2) = 2;
                    volume(3, 3) = 1;

                    op.apply(volume, sino);
                    CHECK_EQ(sino[0], Approx(6));

                    sino[0] = 1;

                    RealVector_t expected(volSize * volSize);

                    expected << 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1;
                    DataContainer resVolume(volumeDescriptor, expected);

                    op.applyAdjoint(sino, volume);
                    CHECK_UNARY(isCwiseApprox(volume, resVolume));
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
            // BinaryMethod op(volumeDescriptor, sinoDescriptor, geom);
            auto range = PlanarDetectorDescriptor(sizeRange, geom);
            auto op = BinaryMethod(volumeDescriptor, range);

            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                IndexVector_t coord(2);
                coord << 0, 2;
                volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 0;
                coord << 0, 3;
                volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 0;
                coord << 1, 3;
                volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 0;
                coord << 2, 3;
                volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 0;

                op.apply(volume, sino);
                DataContainer sZero(sinoDescriptor);
                sZero = 0;
                CHECK_UNARY(isApprox(sino, sZero, epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    coord << 0, 2;
                    volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 1;
                    coord << 0, 3;
                    volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 2;
                    coord << 1, 3;
                    volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 3;
                    coord << 2, 3;
                    volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 4;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0], Approx(10));

                    sino[0] = 1;

                    RealVector_t expected(volSize * volSize);

                    expected << 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0;
                    DataContainer resVolume(volumeDescriptor, expected);

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isCwiseApprox(volume, resVolume));
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
            // BinaryMethod op(volumeDescriptor, sinoDescriptor, geom);
            auto range = PlanarDetectorDescriptor(sizeRange, geom);
            auto op = BinaryMethod(volumeDescriptor, range);

            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                IndexVector_t coord(2);
                coord << 1, 0;
                volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 0;
                coord << 2, 0;
                volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 0;
                coord << 3, 0;
                volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 0;
                coord << 3, 1;
                volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 0;

                op.apply(volume, sino);
                DataContainer sZero(sinoDescriptor);
                sZero = 0;
                CHECK_UNARY(isApprox(sino, sZero, epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    coord << 1, 0;
                    volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 4;
                    coord << 2, 0;
                    volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 3;
                    coord << 3, 0;
                    volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 2;
                    coord << 3, 1;
                    volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 1;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0], Approx(10));

                    sino[0] = 1;

                    RealVector_t expected(volSize * volSize);

                    expected << 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0;
                    DataContainer resVolume(volumeDescriptor, expected);

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isCwiseApprox(volume, resVolume));
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
            // BinaryMethod op(volumeDescriptor, sinoDescriptor, geom);
            auto range = PlanarDetectorDescriptor(sizeRange, geom);
            auto op = BinaryMethod(volumeDescriptor, range);

            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                IndexVector_t coord(2);
                coord << 3, 0;
                volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 0;

                op.apply(volume, sino);
                DataContainer sZero(sinoDescriptor);
                sZero = 0;
                CHECK_UNARY(isApprox(sino, sZero, epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    coord << 3, 0;
                    volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 1;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0], Approx(1));

                    sino[0] = 1;

                    RealVector_t expected(volSize * volSize);
                    expected << 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                    DataContainer resVolume(volumeDescriptor, expected);

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isCwiseApprox(volume, resVolume));
                }
            }
        }
    }

    GIVEN("A 3D setting with a single ray")
    {
        IndexVector_t volumeDims(3), sizeRange(3);
        const index_t volSize = 3;
        const index_t detectorSize = 1;
        const index_t numImgs = 1;
        volumeDims << volSize, volSize, volSize;
        sizeRange << detectorSize, detectorSize, numImgs;
        VolumeDescriptor volumeDescriptor(volumeDims);
        VolumeDescriptor sinoDescriptor(sizeRange);
        DataContainer volume(volumeDescriptor);
        DataContainer sino(sinoDescriptor);

        auto stc = SourceToCenterOfRotation{20 * volSize};
        auto ctr = CenterOfRotationToDetector{volSize};
        auto volData = VolumeData3D{Size3D{volumeDims}};
        auto sinoData = SinogramData3D{Size3D{sizeRange}};

        std::vector<Geometry> geom;

        RealVector_t backProj(volSize * volSize * volSize);

        WHEN("A ray with an angle of 30 degrees goes through the center of the volume")
        {
            // In this case the ray enters and exits the volume along the main direction
            VolumeData3D volDataCopy{volData};
            SinogramData3D sinoDataCopy{sinoData};
            geom.emplace_back(stc, ctr, std::move(volDataCopy), std::move(sinoDataCopy),
                              RotationAngles3D{Gamma{pi_t / 6}});
            // BinaryMethod op(volumeDescriptor, sinoDescriptor, geom);
            auto range = PlanarDetectorDescriptor(sizeRange, geom);
            auto op = BinaryMethod(volumeDescriptor, range);

            THEN("The ray intersects the correct voxels")
            {
                volume = 1;
                IndexVector_t coord(3);
                coord << 1, 1, 1;
                volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 0;
                coord << 2, 1, 0;
                volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 0;
                coord << 1, 1, 0;
                volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 0;
                coord << 0, 1, 2;
                volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 0;
                coord << 1, 1, 2;
                volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 0;

                op.apply(volume, sino);
                REQUIRE_EQ(sino[0], Approx(0));

                AND_THEN("The correct weighting is applied")
                {
                    coord << 1, 1, 1;
                    volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 1;
                    coord << 2, 1, 0;
                    volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 3;
                    coord << 1, 1, 2;
                    volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 2;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0], Approx(6));

                    sino[0] = 1;
                    backProj << 0, 0, 0, 0, 1, 1, 0, 0, 0,

                        0, 0, 0, 0, 1, 0, 0, 0, 0,

                        0, 0, 0, 1, 1, 0, 0, 0, 0;
                    DataContainer resVolume(volumeDescriptor, backProj);

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isCwiseApprox(volume, resVolume));
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
            // BinaryMethod op(volumeDescriptor, sinoDescriptor, geom);
            auto range = PlanarDetectorDescriptor(sizeRange, geom);
            auto op = BinaryMethod(volumeDescriptor, range);

            THEN("The ray intersects the correct voxels")
            {
                volume = 1;
                IndexVector_t coord(3);
                coord << 2, 1, 1;
                volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 0;
                coord << 2, 1, 0;
                volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 0;
                coord << 2, 1, 2;
                volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 0;
                coord << 1, 1, 2;
                volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 0;

                op.apply(volume, sino);
                REQUIRE_EQ(sino[0], Approx(0));

                AND_THEN("The correct weighting is applied")
                {
                    coord << 2, 1, 0;
                    volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 4;
                    coord << 1, 1, 2;
                    volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 3;
                    coord << 2, 1, 1;
                    volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 1;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0], Approx(8));

                    sino[0] = 1;
                    backProj << 0, 0, 0, 0, 0, 1, 0, 0, 0,

                        0, 0, 0, 0, 0, 1, 0, 0, 0,

                        0, 0, 0, 0, 1, 1, 0, 0, 0;
                    DataContainer resVolume(volumeDescriptor, backProj);

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isCwiseApprox(volume, resVolume));
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
            // BinaryMethod op(volumeDescriptor, sinoDescriptor, geom);
            auto range = PlanarDetectorDescriptor(sizeRange, geom);
            auto op = BinaryMethod(volumeDescriptor, range);

            THEN("The ray intersects the correct voxels")
            {
                volume = 1;
                IndexVector_t coord(3);
                coord << 0, 1, 0;
                volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 0;
                coord << 1, 1, 0;
                volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 0;
                coord << 0, 1, 1;
                volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 0;
                coord << 0, 1, 2;
                volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 0;

                op.apply(volume, sino);
                REQUIRE_EQ(sino[0], Approx(0));

                AND_THEN("The correct weighting is applied")
                {
                    coord << 0, 1, 2;
                    volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 4;
                    coord << 1, 1, 0;
                    volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 3;
                    coord << 0, 1, 1;
                    volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 1;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0], Approx(8));

                    sino[0] = 1;
                    backProj << 0, 0, 0, 1, 1, 0, 0, 0, 0,

                        0, 0, 0, 1, 0, 0, 0, 0, 0,

                        0, 0, 0, 1, 0, 0, 0, 0, 0;
                    DataContainer resVolume(volumeDescriptor, backProj);

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isCwiseApprox(volume, resVolume));
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
            // BinaryMethod op(volumeDescriptor, sinoDescriptor, geom);
            auto range = PlanarDetectorDescriptor(sizeRange, geom);
            auto op = BinaryMethod(volumeDescriptor, range);

            THEN("The ray intersects the correct voxels")
            {
                volume = 1;
                IndexVector_t coord(3);
                coord << 0, 1, 0;
                volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 0;

                op.apply(volume, sino);
                REQUIRE_EQ(sino[0], Approx(0));

                AND_THEN("The correct weighting is applied")
                {
                    coord << 0, 1, 0;
                    volume[volumeDescriptor.getIndexFromCoordinate(coord)] = 1;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0], Approx(1));

                    sino[0] = 1;
                    backProj << 0, 0, 0, 1, 0, 0, 0, 0, 0,

                        0, 0, 0, 0, 0, 0, 0, 0, 0,

                        0, 0, 0, 0, 0, 0, 0, 0, 0;
                    DataContainer resVolume(volumeDescriptor, backProj);

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isCwiseApprox(volume, resVolume));
                }
            }
        }
    }
}
