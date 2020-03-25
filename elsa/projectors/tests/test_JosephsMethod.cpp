#include <catch2/catch.hpp>

#include "JosephsMethod.h"
#include "Geometry.h"
#include "Logger.h"
#include "testHelpers.h"

using namespace elsa;

SCENARIO("Testing BinaryVoxelTraversal with only one ray")
{
    // Turn logger of
    Logger::setLevel(Logger::LogLevel::OFF);

    IndexVector_t sizeDomain(2);
    sizeDomain << 5, 5;

    IndexVector_t sizeRange(2);
    sizeRange << 1, 1;

    auto domain = DataDescriptor(sizeDomain);
    auto range = DataDescriptor(sizeRange);

    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "",
                                 " << ", ";");

    GIVEN("A JosephsMethod for 2D and a domain data with all ones")
    {
        std::vector<Geometry> geom;

        auto dataDomain = DataContainer(domain);
        dataDomain = 1;

        auto dataRange = DataContainer(range);
        dataRange = 0;

        WHEN("We have a single ray with 0 degrees")
        {
            geom.emplace_back(100, 5, 0, domain, range);
            auto op = JosephsMethod(domain, range, geom);

            THEN("A^t A x should be close to the original data")
            {
                auto AtAx = DataContainer(domain);

                op.apply(dataDomain, dataRange);

                REQUIRE(dataRange[0] == Approx(5));

                op.applyAdjoint(dataRange, AtAx);

                auto cmp = RealVector_t(sizeDomain.prod());
                cmp << 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0;

                REQUIRE(DataContainer(domain, cmp) == AtAx);
            }
        }

        WHEN("We have a single ray with 180 degrees")
        {
            geom.emplace_back(100, 5, pi_t, domain, range);
            auto op = JosephsMethod(domain, range, geom);

            THEN("A^t A x should be close to the original data")
            {
                auto AtAx = DataContainer(domain);

                op.apply(dataDomain, dataRange);

                REQUIRE(dataRange[0] == Approx(5));

                op.applyAdjoint(dataRange, AtAx);

                auto cmp = RealVector_t(sizeDomain.prod());
                cmp << 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0;

                REQUIRE(isApprox(DataContainer(domain, cmp), AtAx));
            }
        }

        WHEN("We have a single ray with 90 degrees")
        {
            geom.emplace_back(100, 5, pi_t / 2, domain, range);
            auto op = JosephsMethod(domain, range, geom);

            THEN("A^t A x should be close to the original data")
            {
                auto AtAx = DataContainer(domain);

                op.apply(dataDomain, dataRange);

                REQUIRE(dataRange[0] == Approx(5));

                op.applyAdjoint(dataRange, AtAx);

                auto cmp = RealVector_t(sizeDomain.prod());
                cmp << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

                REQUIRE(isApprox(DataContainer(domain, cmp), AtAx));
            }
        }

        WHEN("We have a single ray with 90 degrees")
        {
            geom.emplace_back(100, 5, 3 * pi_t / 2., domain, range);
            auto op = JosephsMethod(domain, range, geom);

            THEN("A^t A x should be close to the original data")
            {
                auto AtAx = DataContainer(domain);

                op.apply(dataDomain, dataRange);

                REQUIRE(dataRange[0] == Approx(5));

                op.applyAdjoint(dataRange, AtAx);

                auto cmp = RealVector_t(sizeDomain.prod());
                cmp << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

                REQUIRE(isApprox(DataContainer(domain, cmp), AtAx));
            }
        }

        WHEN("We have a single ray with 90 degrees")
        {
            geom.emplace_back(100, 5, 45 * pi_t / 180., domain, range);
            auto op = JosephsMethod(domain, range, geom);

            THEN("A^t A x should be close to the original data")
            {
                auto AtAx = DataContainer(domain);

                op.apply(dataDomain, dataRange);
                op.applyAdjoint(dataRange, AtAx);

                auto cmp = RealVector_t(sizeDomain.prod());
                cmp << 10, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0,
                    10;

                REQUIRE(isApprox(DataContainer(domain, cmp), AtAx, epsilon));
            }
        }

        WHEN("We have a single ray with 90 degrees")
        {
            geom.emplace_back(100, 5, 225 * pi_t / 180., domain, range);
            auto op = JosephsMethod(domain, range, geom);

            THEN("A^t A x should be close to the original data")
            {
                auto AtAx = DataContainer(domain);

                op.apply(dataDomain, dataRange);
                op.applyAdjoint(dataRange, AtAx);

                auto cmp = RealVector_t(sizeDomain.prod());
                cmp << 10, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0,
                    10;

                REQUIRE(isApprox(DataContainer(domain, cmp), AtAx, epsilon));
            }
        }
    }
}

SCENARIO("Testing JosephsMethod with only 4 ray")
{
    // Turn logger of
    Logger::setLevel(Logger::LogLevel::OFF);

    IndexVector_t sizeDomain(2);
    sizeDomain << 5, 5;

    IndexVector_t sizeRange(2);
    sizeRange << 1, 4;

    auto domain = DataDescriptor(sizeDomain);
    auto range = DataDescriptor(sizeRange);

    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "",
                                 " << ", ";");

    GIVEN("A JosephsMethod for 2D and a domain data with all ones")
    {
        std::vector<Geometry> geom;

        auto dataDomain = DataContainer(domain);
        dataDomain = 1;

        auto dataRange = DataContainer(range);
        dataRange = 0;

        WHEN("We have a single ray with 0, 90, 180, 270 degrees")
        {
            geom.emplace_back(100, 5, 0, domain, range);
            geom.emplace_back(100, 5, pi_t / 2, domain, range);
            geom.emplace_back(100, 5, pi_t, domain, range);
            geom.emplace_back(100, 5, 3 * pi_t / 2, domain, range);
            auto op = JosephsMethod(domain, range, geom);

            THEN("A^t A x should be close to the original data")
            {
                auto AtAx = DataContainer(domain);

                op.apply(dataDomain, dataRange);
                op.applyAdjoint(dataRange, AtAx);

                auto cmp = RealVector_t(sizeDomain.prod());
                cmp << 0, 0, 10, 0, 0, 0, 0, 10, 0, 0, 10, 10, 20, 10, 10, 0, 0, 10, 0, 0, 0, 0, 10,
                    0, 0;

                REQUIRE(isApprox(DataContainer(domain, cmp), AtAx));
            }
        }
    }
}

SCENARIO("Calls to functions of super class")
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
        DataDescriptor volumeDescriptor(volumeDims);
        DataDescriptor sinoDescriptor(sinoDims);
        DataContainer volume(volumeDescriptor);
        volume = 0;
        DataContainer sino(sinoDescriptor);
        sino = 0;
        std::vector<Geometry> geom;
        for (std::size_t i = 0; i < numImgs; i++) {
            real_t angle = static_cast<real_t>(i * 2) * pi_t / 50;
            geom.emplace_back(20 * volSize, volSize, angle, volumeDescriptor, sinoDescriptor);
        }

        JosephsMethod op(volumeDescriptor, sinoDescriptor, geom);

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
                REQUIRE(isApprox(sino, sinoClone));

                op.applyAdjoint(sino, volume);
                opClone->applyAdjoint(sino, volumeClone);

                DataContainer resultsDifference = volume - volumeClone;
                REQUIRE(resultsDifference.squaredL2Norm() == Approx(0.0).margin(1e-5));
            }
        }
    }
}

SCENARIO("Output DataContainer is not zero initialized")
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
        DataDescriptor volumeDescriptor(volumeDims);
        DataDescriptor sinoDescriptor(sinoDims);
        DataContainer volume(volumeDescriptor);
        DataContainer sino(sinoDescriptor);
        std::vector<Geometry> geom;
        geom.emplace_back(20 * volSize, volSize, 0.0, volumeDescriptor, sinoDescriptor);
        JosephsMethod op(volumeDescriptor, sinoDescriptor, geom,
                         JosephsMethod<>::Interpolation::LINEAR);

        WHEN("Sinogram conatainer is not zero initialized and we project through an empty volume")
        {
            volume = 0;
            sino = 1;

            THEN("Result is zero")
            {
                op.apply(volume, sino);

                DataContainer zero(sinoDescriptor);
                zero = 0;
                REQUIRE(isApprox(sino, zero, epsilon));
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
                REQUIRE(isApprox(volume, zero, epsilon));
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
        DataDescriptor volumeDescriptor(volumeDims);
        DataDescriptor sinoDescriptor(sinoDims);
        DataContainer volume(volumeDescriptor);
        DataContainer sino(sinoDescriptor);
        std::vector<Geometry> geom;

        geom.emplace_back(volSize * 20, volSize, volumeDescriptor, sinoDescriptor, 0);
        JosephsMethod op(volumeDescriptor, sinoDescriptor, geom,
                         JosephsMethod<>::Interpolation::LINEAR);

        WHEN("Sinogram conatainer is not zero initialized and we project through an empty volume")
        {
            volume = 0;
            sino = 1;

            THEN("Result is zero")
            {
                op.apply(volume, sino);
                DataContainer zero(sinoDescriptor);
                zero = 0;
                REQUIRE(isApprox(sino, zero, epsilon));
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
                REQUIRE(isApprox(volume, zero, epsilon));
            }
        }
    }
}

SCENARIO("Rays not intersecting the bounding box are present")
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
        DataDescriptor volumeDescriptor(volumeDims);
        DataDescriptor sinoDescriptor(sinoDims);
        DataContainer volume(volumeDescriptor);
        DataContainer sino(sinoDescriptor);
        volume = 1;
        sino = 1;
        std::vector<Geometry> geom;

        WHEN("Tracing along a y-axis-aligned ray with a negative x-coordinate of origin")
        {
            geom.emplace_back(20 * volSize, volSize, 0.0, volumeDescriptor, sinoDescriptor, 0.0,
                              volSize);

            JosephsMethod op(volumeDescriptor, sinoDescriptor, geom,
                             JosephsMethod<>::Interpolation::LINEAR);

            THEN("Result of forward projection is zero")
            {
                op.apply(volume, sino);
                DataContainer zero(sinoDescriptor);
                zero = 0;
                REQUIRE(isApprox(sino, zero, epsilon));

                AND_THEN("Result of backprojection is zero")
                {
                    op.applyAdjoint(sino, volume);
                    DataContainer zero(volumeDescriptor);
                    zero = 0;
                    REQUIRE(isApprox(volume, zero, epsilon));
                }
            }
        }

        WHEN("Tracing along a y-axis-aligned ray with a x-coordinate of origin beyond the bounding "
             "box")
        {
            geom.emplace_back(20 * volSize, volSize, 0.0, volumeDescriptor, sinoDescriptor, 0.0,
                              -volSize);

            JosephsMethod op(volumeDescriptor, sinoDescriptor, geom,
                             JosephsMethod<>::Interpolation::LINEAR);

            THEN("Result of forward projection is zero")
            {
                op.apply(volume, sino);
                DataContainer zero(sinoDescriptor);
                zero = 0;
                REQUIRE(isApprox(sino, zero, epsilon));

                AND_THEN("Result of backprojection is zero")
                {
                    op.applyAdjoint(sino, volume);
                    DataContainer zero(volumeDescriptor);
                    zero = 0;
                    REQUIRE(isApprox(volume, zero, epsilon));
                }
            }
        }

        WHEN("Tracing along a x-axis-aligned ray with a negative y-coordinate of origin")
        {
            geom.emplace_back(20 * volSize, volSize, pi_t / 2, volumeDescriptor, sinoDescriptor,
                              0.0, 0.0, volSize);

            JosephsMethod op(volumeDescriptor, sinoDescriptor, geom,
                             JosephsMethod<>::Interpolation::LINEAR);

            THEN("Result of forward projection is zero")
            {
                op.apply(volume, sino);
                DataContainer zero(sinoDescriptor);
                zero = 0;
                REQUIRE(isApprox(sino, zero, epsilon));

                AND_THEN("Result of backprojection is zero")
                {
                    op.applyAdjoint(sino, volume);
                    DataContainer zero(volumeDescriptor);
                    zero = 0;
                    REQUIRE(isApprox(volume, zero, epsilon));
                }
            }
        }

        WHEN("Tracing along a x-axis-aligned ray with a y-coordinate of origin beyond the bounding "
             "box")
        {
            geom.emplace_back(20 * volSize, volSize, pi_t / 2, volumeDescriptor, sinoDescriptor,
                              0.0, 0.0, -volSize);

            JosephsMethod op(volumeDescriptor, sinoDescriptor, geom,
                             JosephsMethod<>::Interpolation::LINEAR);

            THEN("Result of forward projection is zero")
            {
                op.apply(volume, sino);
                DataContainer zero(sinoDescriptor);
                zero = 0;
                REQUIRE(isApprox(sino, zero, epsilon));

                AND_THEN("Result of backprojection is zero")
                {
                    op.applyAdjoint(sino, volume);
                    DataContainer zero(volumeDescriptor);
                    zero = 0;
                    REQUIRE(isApprox(volume, zero, epsilon));
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
        DataDescriptor volumeDescriptor(volumeDims);
        DataDescriptor sinoDescriptor(sinoDims);
        DataContainer volume(volumeDescriptor);
        DataContainer sino(sinoDescriptor);
        volume = 1;
        sino = 1;
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
            WHEN("Tracing along a " + ali[i] + "-axis-aligned ray with negative " + neg[i]
                 + "-coodinate of origin")
            {
                geom.emplace_back(20 * volSize, volSize, volumeDescriptor, sinoDescriptor, gamma[i],
                                  beta[i], alpha[i], 0.0, 0.0, offsetx[i], offsety[i], offsetz[i]);

                JosephsMethod op(volumeDescriptor, sinoDescriptor, geom,
                                 JosephsMethod<>::Interpolation::LINEAR);

                THEN("Result of forward projection is zero")
                {
                    op.apply(volume, sino);
                    DataContainer zero(sinoDescriptor);
                    zero = 0;
                    REQUIRE(isApprox(sino, zero, epsilon));

                    AND_THEN("Result of backprojection is zero")
                    {
                        op.applyAdjoint(sino, volume);
                        DataContainer zero(volumeDescriptor);
                        zero = 0;
                        REQUIRE(isApprox(volume, zero, epsilon));
                    }
                }
            }
        }
    }
}

SCENARIO("Axis-aligned rays are present")
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
        DataDescriptor volumeDescriptor(volumeDims);
        DataDescriptor sinoDescriptor(sinoDims);
        DataContainer volume(volumeDescriptor);
        DataContainer sino(sinoDescriptor);
        std::vector<Geometry> geom;

        const index_t numCases = 4;
        const real_t angles[numCases] = {0.0, pi_t / 2, pi_t, 3 * pi_t / 2};
        RealVector_t backProj[2];
        backProj[0].resize(volSize * volSize);
        backProj[1].resize(volSize * volSize);
        backProj[1] << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

        backProj[0] << 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0;

        for (index_t i = 0; i < numCases; i++) {
            WHEN("An axis-aligned ray with an angle of " + std::to_string(angles[i])
                 + " radians passes through the center of a pixel")
            {
                geom.emplace_back(volSize * 20, volSize, angles[i], volumeDescriptor,
                                  sinoDescriptor);

                JosephsMethod op(volumeDescriptor, sinoDescriptor, geom,
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
                        REQUIRE(sino[0] == Approx(1));
                    }

                    AND_THEN("The backprojection sets the values of all hit pixels to the detector "
                             "value")
                    {
                        op.applyAdjoint(sino, volume);
                        REQUIRE(isApprox(volume, DataContainer(volumeDescriptor, backProj[i % 2])));
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
            WHEN("An axis-aligned ray with an angle of " + std::to_string(angles[i])
                 + " radians does not pass through the center of a pixel")
            {
                geom.emplace_back(volSize * 2000, volSize, angles[i], volumeDescriptor,
                                  sinoDescriptor, 0.0, -offsetx[i], -offsety[i]);
                JosephsMethod op(volumeDescriptor, sinoDescriptor, geom,
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
                        REQUIRE(sino[0] == Approx(0.75));
                    }

                    AND_THEN("The backprojection yields the exact adjoint")
                    {
                        sino[0] = 1;
                        op.applyAdjoint(sino, volume);
                        REQUIRE(isApprox(volume, DataContainer(volumeDescriptor, backProj[i % 2])));
                    }
                }
            }
        }

        backProj[0] << 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;

        WHEN("A y-axis-aligned ray runs along the right volume boundary")
        {
            geom.emplace_back(volSize * 2000, volSize, 0.0, volumeDescriptor, sinoDescriptor, 0.0,
                              (volSize * 0.5));
            JosephsMethod op(volumeDescriptor, sinoDescriptor, geom);

            THEN("The result of projecting through a pixel is exactly the pixel's value (we mirror "
                 "values at the border for the purpose of interpolation)")
            {
                for (index_t j = 0; j < volSize; j++) {
                    volume = 0;
                    volume(volSize - 1, j) = 1;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == 1);
                }

                AND_THEN(
                    "The slow backprojection yields the exact adjoint, the fast backprojection "
                    "also yields the exact adjoint for a very distant x-ray source")
                {
                    sino[0] = 1;
                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer(volumeDescriptor, backProj[0])));
                }
            }
        }

        backProj[0] << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0;

        WHEN("A y-axis-aligned ray runs along the left volume boundary")
        {
            geom.emplace_back(volSize * 2000, volSize, 0.0, volumeDescriptor, sinoDescriptor, 0.0,
                              -volSize / 2.0);
            JosephsMethod op(volumeDescriptor, sinoDescriptor, geom);

            THEN("The result of projecting through a pixel is exactly the pixel's value (we mirror "
                 "values at the border for the purpose of interpolation)")
            {
                for (index_t j = 0; j < volSize; j++) {
                    volume = 0;
                    volume(0, j) = 1;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == 1);
                }

                AND_THEN(
                    "The slow backprojection yields the exact adjoint, the fast backprojection "
                    "also yields the exact adjoint for a very distant x-ray source")
                {
                    sino[0] = 1;
                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer(volumeDescriptor, backProj[0])));
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
        DataDescriptor volumeDescriptor(volumeDims);
        DataDescriptor sinoDescriptor(sinoDims);
        DataContainer volume(volumeDescriptor);
        DataContainer sino(sinoDescriptor);
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
            WHEN("A " + al[i] + "-axis-aligned ray passes through the center of a pixel")
            {
                geom.emplace_back(volSize * 20, volSize, volumeDescriptor, sinoDescriptor, gamma[i],
                                  beta[i]);
                JosephsMethod op(volumeDescriptor, sinoDescriptor, geom,
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
                        REQUIRE(sino[0] == Approx(1.0));
                    }

                    AND_THEN("The backprojection sets the values of all hit voxels to the detector "
                             "value")
                    {
                        op.applyAdjoint(sino, volume);
                        REQUIRE(isApprox(volume, DataContainer(volumeDescriptor, backProj[i / 2])));
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
            WHEN("A " + al[i] + "-axis-aligned ray does not pass through the center of a voxel")
            {
                geom.emplace_back(volSize * 2000, volSize, volumeDescriptor, sinoDescriptor,
                                  gamma[i], beta[i], 0.0, 0.0, 0.0, -offsetx[i], -offsety[i],
                                  -offsetz[i]);
                JosephsMethod op(volumeDescriptor, sinoDescriptor, geom,
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
                        REQUIRE(sino[0] == Approx(0.75));
                    }

                    AND_THEN("The backprojection yields the exact adjoint")
                    {
                        sino[0] = 1;

                        op.applyAdjoint(sino, volume);
                        REQUIRE(isApprox(volume, DataContainer(volumeDescriptor, backProj[i / 2])));
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
            WHEN("A z-axis-aligned ray runs along the " + al[i] + " of the volume")
            {
                // x-ray source must be very far from the volume center to make testing of the fast
                // backprojection simpler
                geom.emplace_back(volSize * 2000, volSize, volumeDescriptor, sinoDescriptor, 0.0,
                                  0.0, 0.0, 0.0, 0.0, -offsetx[i], -offsety[i]);
                JosephsMethod op(volumeDescriptor, sinoDescriptor, geom);
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
                        REQUIRE(sino[0] == 1);
                    }

                    AND_THEN("The backprojection yields the exact adjoint")
                    {
                        sino[0] = 1;

                        op.applyAdjoint(sino, volume);
                        REQUIRE(isApprox(volume, DataContainer(volumeDescriptor, backProj[i])));
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
        DataDescriptor volumeDescriptor(volumeDims);
        DataDescriptor sinoDescriptor(sinoDims);
        DataContainer volume(volumeDescriptor);
        DataContainer sino(sinoDescriptor);
        std::vector<Geometry> geom;

        WHEN("x-, y and z-axis-aligned rays are present")
        {
            real_t beta[numImgs] = {0.0, 0.0, 0.0, 0.0, pi_t / 2, 3 * pi_t / 2};
            real_t gamma[numImgs] = {0.0, pi_t, pi_t / 2, 3 * pi_t / 2, pi_t / 2, 3 * pi_t / 2};

            for (index_t i = 0; i < numImgs; i++)
                geom.emplace_back(volSize * 20, volSize, volumeDescriptor, sinoDescriptor, gamma[i],
                                  beta[i]);

            JosephsMethod op(volumeDescriptor, sinoDescriptor, geom,
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
                    REQUIRE(sino[i] == Approx(3.0));

                AND_THEN("Backprojections yield the exact adjoint")
                {
                    RealVector_t cmp(volSize * volSize * volSize);

                    cmp << 0, 0, 0, 0, 6, 0, 0, 0, 0,

                        0, 6, 0, 6, 18, 6, 0, 6, 0,

                        0, 0, 0, 0, 6, 0, 0, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer(volumeDescriptor, cmp)));
                }
            }
        }
    }
}

SCENARIO("Projection under an angle")
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
        DataDescriptor volumeDescriptor(volumeDims);
        DataDescriptor sinoDescriptor(sinoDims);
        DataContainer volume(volumeDescriptor);
        DataContainer sino(sinoDescriptor);
        std::vector<Geometry> geom;

        WHEN("Projecting under an angle of 30 degrees and ray goes through center of volume")
        {
            // In this case the ray enters and exits the volume through the borders along the main
            // direction Weighting for all interpolated values should be the same
            geom.emplace_back(volSize * 20, volSize, -pi_t / 6, volumeDescriptor, sinoDescriptor);
            JosephsMethod op(volumeDescriptor, sinoDescriptor, geom,
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
                CHECK(isApprox(sino, zero, epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    volume(3, 0) = 1;
                    volume(1, 1) = 1;
                    volume(1, 2) = 1;
                    volume(1, 3) = 1;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == Approx(2 * weight));

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
                    REQUIRE(isApprox(volume, DataContainer(volumeDescriptor, opExpected)));
                }
            }
        }

        WHEN("Projecting under an angle of 30 degrees and ray enters through the right border")
        {
            // In this case the ray exits through a border along the main ray direction, but enters
            // through a border not along the main direction First pixel should be weighted
            // differently
            geom.emplace_back(volSize * 20, volSize, -pi_t / 6, volumeDescriptor, sinoDescriptor,
                              0.0, std::sqrt(3.f));
            JosephsMethod op(volumeDescriptor, sinoDescriptor, geom,
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
                CHECK(isApprox(sino, zero, epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    volume(3, 1) = 1;
                    volume(2, 2) = 1;
                    volume(2, 3) = 1;

                    op.apply(volume, sino);
                    REQUIRE(sino[0]
                            == Approx((4 - 2 * std::sqrt(3.f)) * (std::sqrt(3.f) - 1)
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
                    REQUIRE(isApprox(volume, DataContainer(volumeDescriptor, opExpected), epsilon));
                }
            }
        }

        WHEN("Projecting under an angle of 30 degrees and ray exits through the left border")
        {
            // In this case the ray enters through a border along the main ray direction, but exits
            // through a border not along the main direction Last pixel should be weighted
            // differently
            geom.emplace_back(volSize * 20, volSize, -pi_t / 6, volumeDescriptor, sinoDescriptor,
                              0.0, -std::sqrt(3.f));
            JosephsMethod op(volumeDescriptor, sinoDescriptor, geom,
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
                CHECK(isApprox(sino, zero, epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    volume(1, 0) = 1;
                    volume(0, 1) = 1;

                    op.apply(volume, sino);
                    REQUIRE(sino[0]
                            == Approx((std::sqrt(3.f) - 1) + (5.0 / 3.0 - 1 / std::sqrt(3.f))
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
                    REQUIRE(isApprox(volume, DataContainer(volumeDescriptor, opExpected)));
                }
            }
        }

        WHEN("Projecting under an angle of 30 degrees and ray only intersects a single pixel")
        {
            // This is a special case that is handled separately in both forward and backprojection
            geom.emplace_back(volSize * 20, volSize, -pi_t / 6, volumeDescriptor, sinoDescriptor,
                              0.0, -2 - std::sqrt(3.f) / 2);
            JosephsMethod op(volumeDescriptor, sinoDescriptor, geom,
                             JosephsMethod<>::Interpolation::LINEAR);

            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                volume(0, 0) = 0;

                op.apply(volume, sino);
                DataContainer zero(sinoDescriptor);
                zero = 0;
                CHECK(isApprox(sino, zero, epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    volume(0, 0) = 1;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == Approx(1 / std::sqrt(3.f)).epsilon(epsilon));

                    sino[0] = 1;

                    RealVector_t opExpected(volSize * volSize);
                    opExpected << 1 / std::sqrt(3.f), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer(volumeDescriptor, opExpected)));
                }
            }
        }

        WHEN("Projecting under an angle of 120 degrees and ray goes through center of volume")
        {
            // In this case the ray enters and exits the volume through the borders along the main
            // direction Weighting for all interpolated values should be the same
            geom.emplace_back(volSize * 20, volSize, -2 * pi_t / 3, volumeDescriptor,
                              sinoDescriptor);
            JosephsMethod op(volumeDescriptor, sinoDescriptor, geom,
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
                CHECK(isApprox(sino, zero, epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    volume(3, 3) = 1;
                    volume(0, 1) = 1;
                    volume(1, 1) = 1;
                    volume(2, 1) = 1;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == Approx(2 * weight));

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
                    REQUIRE(isApprox(volume, DataContainer(volumeDescriptor, opExpected)));
                }
            }
        }

        WHEN("Projecting under an angle of 120 degrees and ray enters through the top border")
        {
            // In this case the ray exits through a border along the main ray direction, but enters
            // through a border not along the main direction First pixel should be weighted
            // differently
            geom.emplace_back(volSize * 20, volSize, -2 * pi_t / 3, volumeDescriptor,
                              sinoDescriptor, 0.0, 0.0, std::sqrt(3.f));
            JosephsMethod op(volumeDescriptor, sinoDescriptor, geom,
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
                CHECK(isApprox(sino, zero, epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    volume(2, 3) = 1;
                    volume(1, 2) = 1;
                    volume(1, 3) = 1;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == Approx((4 - 2 * std::sqrt(3.f)) + (2 / std::sqrt(3.f))));

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
                    REQUIRE(isApprox(volume, DataContainer(volumeDescriptor, opExpected), epsilon));
                }
            }
        }

        WHEN("Projecting under an angle of 120 degrees and ray exits through the bottom border")
        {
            // In this case the ray enters through a border along the main ray direction, but exits
            // through a border not along the main direction Last pixel should be weighted
            // differently
            geom.emplace_back(volSize * 20, volSize, -2 * pi_t / 3, volumeDescriptor,
                              sinoDescriptor, 0.0, 0.0, -std::sqrt(3.f));
            JosephsMethod op(volumeDescriptor, sinoDescriptor, geom,
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
                CHECK(isApprox(sino, zero, epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    volume(2, 0) = 1;
                    volume(3, 1) = 1;

                    op.apply(volume, sino);
                    REQUIRE(sino[0]
                            == Approx((std::sqrt(3.f) - 1) + (5.0 / 3.0 - 1 / std::sqrt(3.f))
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
                    REQUIRE(isApprox(volume, DataContainer(volumeDescriptor, opExpected)));
                }
            }
        }

        WHEN("Projecting under an angle of 120 degrees and ray only intersects a single pixel")
        {
            // This is a special case that is handled separately in both forward and backprojection
            geom.emplace_back(volSize * 20, volSize, -2 * pi_t / 3, volumeDescriptor,
                              sinoDescriptor, 0.0, 0.0, -2 - std::sqrt(3.f) / 2);
            JosephsMethod op(volumeDescriptor, sinoDescriptor, geom,
                             JosephsMethod<>::Interpolation::LINEAR);

            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                volume(3, 0) = 0;

                op.apply(volume, sino);
                DataContainer zero(sinoDescriptor);
                zero = 0;
                CHECK(isApprox(sino, zero, epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    volume(3, 0) = 1;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == Approx(1 / std::sqrt(3.f)).epsilon(epsilon));

                    sino[0] = 1;

                    RealVector_t opExpected(volSize * volSize);
                    opExpected << 0, 0, 0, 1 / std::sqrt(3.f), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer(volumeDescriptor, opExpected), epsilon));
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
        DataDescriptor volumeDescriptor(volumeDims);
        DataDescriptor sinoDescriptor(sinoDims);
        DataContainer volume(volumeDescriptor);
        DataContainer sino(sinoDescriptor);
        std::vector<Geometry> geom;

        RealVector_t backProj(volSize * volSize * volSize);

        WHEN("A ray with an angle of 30 degrees goes through the center of the volume")
        {
            // In this case the ray enters and exits the volume along the main direction
            geom.emplace_back(volSize * 20, volSize, volumeDescriptor, sinoDescriptor, pi_t / 6);
            JosephsMethod op(volumeDescriptor, sinoDescriptor, geom,
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
                REQUIRE(sino[0] == Approx(0).margin(1e-5));

                AND_THEN("The correct weighting is applied")
                {
                    volume(1, 1, 1) = 1;
                    volume(2, 1, 0) = 3;
                    volume(1, 1, 2) = 2;

                    op.apply(volume, sino);
                    REQUIRE(sino[0]
                            == Approx(2 / std::sqrt(3.f) + 2 - 4.0f / 3 + 4 / std::sqrt(3.f)));

                    sino[0] = 1;
                    backProj << 0, 0, 0, 0, 2 / std::sqrt(3.f) - 2.0f / 3, 2.0f / 3, 0, 0, 0,

                        0, 0, 0, 0, 2 / std::sqrt(3.f), 0, 0, 0, 0,

                        0, 0, 0, 2.0f / 3, 2 / std::sqrt(3.f) - 2.0f / 3, 0, 0, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer(volumeDescriptor, backProj)));
                }
            }
        }

        WHEN("A ray with an angle of 30 degrees enters through the right border")
        {
            // In this case the ray enters through a border orthogonal to a non-main direction
            geom.emplace_back(volSize * 20, volSize, volumeDescriptor, sinoDescriptor, pi_t / 6,
                              0.0, 0.0, 0.0, 0.0, 1);
            JosephsMethod op(volumeDescriptor, sinoDescriptor, geom,
                             JosephsMethod<>::Interpolation::LINEAR);

            THEN("The ray intersects the correct voxels")
            {
                volume = 1;
                volume(2, 1, 1) = 0;
                volume(2, 1, 0) = 0;
                volume(2, 1, 2) = 0;
                volume(1, 1, 2) = 0;

                op.apply(volume, sino);
                REQUIRE(sino[0] == Approx(0).margin(1e-5));

                AND_THEN("The correct weighting is applied")
                {
                    volume(2, 1, 0) = 4;
                    volume(1, 1, 2) = 3;
                    volume(2, 1, 1) = 1;

                    op.apply(volume, sino);
                    REQUIRE(sino[0]
                            == Approx((std::sqrt(3.f) + 1) * (1 - 1 / std::sqrt(3.f)) + 3
                                      - std::sqrt(3.f) / 2 + 2 / std::sqrt(3.f))
                                   .epsilon(epsilon));

                    sino[0] = 1;
                    backProj << 0, 0, 0, 0, 0,
                        ((std::sqrt(3.f) + 1) / 4) * (1 - 1 / std::sqrt(3.f)), 0, 0, 0,

                        0, 0, 0, 0, 0, 2 / std::sqrt(3.f) + 1 - std::sqrt(3.f) / 2, 0, 0, 0,

                        0, 0, 0, 0, 2.0f / 3, 2 / std::sqrt(3.f) - 2.0f / 3, 0, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer(volumeDescriptor, backProj)));
                }
            }
        }

        WHEN("A ray with an angle of 30 degrees exits through the left border")
        {
            // In this case the ray exit through a border orthogonal to a non-main direction
            geom.emplace_back(volSize * 20, volSize, volumeDescriptor, sinoDescriptor, pi_t / 6,
                              0.0, 0.0, 0.0, 0.0, -1);
            JosephsMethod op(volumeDescriptor, sinoDescriptor, geom,
                             JosephsMethod<>::Interpolation::LINEAR);

            THEN("The ray intersects the correct voxels")
            {
                volume = 1;
                volume(0, 1, 0) = 0;
                volume(1, 1, 0) = 0;
                volume(0, 1, 1) = 0;
                volume(0, 1, 2) = 0;

                op.apply(volume, sino);
                REQUIRE(sino[0] == Approx(0).margin(1e-5));

                AND_THEN("The correct weighting is applied")
                {
                    volume(0, 1, 2) = 4;
                    volume(1, 1, 0) = 3;
                    volume(0, 1, 1) = 1;

                    op.apply(volume, sino);
                    REQUIRE(sino[0]
                            == Approx((std::sqrt(3.f) + 1) * (1 - 1 / std::sqrt(3.f)) + 3
                                      - std::sqrt(3.f) / 2 + 2 / std::sqrt(3.f))
                                   .epsilon(epsilon));

                    sino[0] = 1;
                    backProj << 0, 0, 0, 2 / std::sqrt(3.f) - 2.0f / 3, 2.0f / 3, 0, 0, 0, 0,

                        0, 0, 0, 2 / std::sqrt(3.f) + 1 - std::sqrt(3.f) / 2, 0, 0, 0, 0, 0,

                        0, 0, 0, ((std::sqrt(3.f) + 1) / 4) * (1 - 1 / std::sqrt(3.f)), 0, 0, 0, 0,
                        0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer(volumeDescriptor, backProj)));
                }
            }
        }

        WHEN("A ray with an angle of 30 degrees only intersects a single voxel")
        {
            // special case - no interior voxels, entry and exit voxels are the same
            geom.emplace_back(volSize * 20, volSize, volumeDescriptor, sinoDescriptor, pi_t / 6,
                              0.0, 0.0, 0.0, 0.0, -2);
            JosephsMethod op(volumeDescriptor, sinoDescriptor, geom,
                             JosephsMethod<>::Interpolation::LINEAR);

            THEN("The ray intersects the correct voxels")
            {
                volume = 1;
                volume(0, 1, 0) = 0;

                op.apply(volume, sino);
                REQUIRE(sino[0] == Approx(0).margin(1e-5));

                AND_THEN("The correct weighting is applied")
                {
                    volume(0, 1, 0) = 1;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == Approx(std::sqrt(3.f) - 1).epsilon(epsilon));

                    sino[0] = 1;
                    backProj << 0, 0, 0, std::sqrt(3.f) - 1, 0, 0, 0, 0, 0,

                        0, 0, 0, 0, 0, 0, 0, 0, 0,

                        0, 0, 0, 0, 0, 0, 0, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer(volumeDescriptor, backProj), epsilon));
                }
            }
        }
    }
}
