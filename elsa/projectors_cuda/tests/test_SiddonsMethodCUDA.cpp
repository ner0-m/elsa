#include <catch2/catch.hpp>

#include "SiddonsMethodCUDA.h"
#include "SiddonsMethod.h"
#include "Geometry.h"
#include "testHelpers.h"
#include "Logger.h"

#include <array>

using namespace elsa;

/*
 * this function declaration can be used in conjunction with decltype to deduce the
 * template parameter of a templated class at compile time
 *
 * the template parameter must be a typename
 */
template <template <typename> typename T, typename data_t>
constexpr data_t return_data_t(const T<data_t>&);

TEMPLATE_TEST_CASE("Scenario: Calls to functions of super class", "", SiddonsMethodCUDA<float>,
                   SiddonsMethodCUDA<double>, SiddonsMethod<float>, SiddonsMethod<double>)
{
    // Turn logger of
    Logger::setLevel(Logger::LogLevel::OFF);

    using data_t = decltype(return_data_t(std::declval<TestType>()));
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
        DataContainer<data_t> volume(volumeDescriptor);
        volume = 1;
        DataContainer<data_t> sino(sinoDescriptor);
        std::vector<Geometry> geom;
        for (index_t i = 0; i < numImgs; i++) {
            real_t angle = static_cast<real_t>(i) * 2 * pi_t / 50;
            geom.emplace_back(20 * volSize, volSize, angle, volumeDescriptor, sinoDescriptor);
        }
        TestType op(volumeDescriptor, sinoDescriptor, geom);

        WHEN("Projector is cloned")
        {
            auto opClone = op.clone();
            auto sinoClone = DataContainer<data_t>(sinoDescriptor);
            auto volumeClone = DataContainer<data_t>(volumeDescriptor);

            THEN("Results do not change (may still be slightly different due to summation being "
                 "performed in a different order)")
            {
                op.apply(volume, sino);
                opClone->apply(volume, sinoClone);
                REQUIRE(isApprox<data_t>(sino, sinoClone, epsilon));

                op.applyAdjoint(sino, volume);
                opClone->applyAdjoint(sino, volumeClone);
                REQUIRE(isApprox<data_t>(volume, volumeClone, epsilon));
            }
        }
    }
}

TEMPLATE_TEST_CASE("Scenario: Output DataContainer<data_t> is not zero initialized", "",
                   SiddonsMethodCUDA<float>, SiddonsMethodCUDA<double>, SiddonsMethod<float>,
                   SiddonsMethod<double>)
{
    // Turn logger of
    Logger::setLevel(Logger::LogLevel::OFF);

    using data_t = decltype(return_data_t(std::declval<TestType>()));
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
        DataContainer<data_t> volume(volumeDescriptor);
        DataContainer<data_t> sino(sinoDescriptor);
        std::vector<Geometry> geom;
        geom.emplace_back(20 * volSize, volSize, 0.0, volumeDescriptor, sinoDescriptor);
        TestType op(volumeDescriptor, sinoDescriptor, geom);

        WHEN("Sinogram conatainer is not zero initialized and we project through an empty volume")
        {
            volume = 0;
            sino = 1;

            THEN("Result is zero")
            {
                op.apply(volume, sino);
                DataContainer<data_t> zero(sinoDescriptor);
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
                DataContainer<data_t> zero(volumeDescriptor);
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
        DataContainer<data_t> volume(volumeDescriptor);
        DataContainer<data_t> sino(sinoDescriptor);
        std::vector<Geometry> geom;

        geom.emplace_back(volSize * 20, volSize, volumeDescriptor, sinoDescriptor, 0);
        TestType op(volumeDescriptor, sinoDescriptor, geom);

        WHEN("Sinogram conatainer is not zero initialized and we project through an empty volume")
        {
            volume = 0;
            sino = 1;

            THEN("Result is zero")
            {
                op.apply(volume, sino);
                DataContainer<data_t> zero(sinoDescriptor);
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
                DataContainer<data_t> zero(volumeDescriptor);
                zero = 0;
                REQUIRE(isApprox(volume, zero, epsilon));
            }
        }
    }
}

TEMPLATE_TEST_CASE("Scenario: Rays not intersecting the bounding box are present", "",
                   SiddonsMethodCUDA<float>, SiddonsMethodCUDA<double>, SiddonsMethod<float>,
                   SiddonsMethod<double>)
{
    // Turn logger of
    Logger::setLevel(Logger::LogLevel::OFF);

    using data_t = decltype(return_data_t(std::declval<TestType>()));
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
        DataContainer<data_t> volume(volumeDescriptor);
        DataContainer<data_t> sino(sinoDescriptor);
        volume = 1;
        sino = 1;
        std::vector<Geometry> geom;

        WHEN("Tracing along a y-axis-aligned ray with a negative x-coordinate of origin")
        {
            geom.emplace_back(20 * volSize, volSize, 0.0, volumeDescriptor, sinoDescriptor, 0.0,
                              volSize);

            TestType op(volumeDescriptor, sinoDescriptor, geom);

            THEN("Result of forward projection is zero")
            {
                op.apply(volume, sino);
                DataContainer<data_t> zero(sinoDescriptor);
                zero = 0;
                REQUIRE(isApprox(sino, zero, epsilon));

                AND_THEN("Result of backprojection is zero")
                {
                    op.applyAdjoint(sino, volume);
                    DataContainer<data_t> zero(volumeDescriptor);
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

            TestType op(volumeDescriptor, sinoDescriptor, geom);

            THEN("Result of forward projection is zero")
            {
                op.apply(volume, sino);
                DataContainer<data_t> zero(sinoDescriptor);
                zero = 0;
                REQUIRE(isApprox(sino, zero, epsilon));

                AND_THEN("Result of backprojection is zero")
                {
                    op.applyAdjoint(sino, volume);
                    DataContainer<data_t> zero(volumeDescriptor);
                    zero = 0;
                    REQUIRE(isApprox(volume, zero, epsilon));
                }
            }
        }

        WHEN("Tracing along a x-axis-aligned ray with a negative y-coordinate of origin")
        {
            geom.emplace_back(20 * volSize, volSize, pi_t / 2, volumeDescriptor, sinoDescriptor,
                              0.0, 0.0, volSize);

            TestType op(volumeDescriptor, sinoDescriptor, geom);

            THEN("Result of forward projection is zero")
            {
                op.apply(volume, sino);
                DataContainer<data_t> zero(sinoDescriptor);
                zero = 0;
                CHECK(isApprox(sino, zero, epsilon));

                AND_THEN("Result of backprojection is zero")
                {
                    op.applyAdjoint(sino, volume);
                    DataContainer<data_t> zero(volumeDescriptor);
                    zero = 0;
                    CHECK(isApprox(volume, zero, epsilon));
                }
            }
        }

        WHEN("Tracing along a x-axis-aligned ray with a y-coordinate of origin beyond the bounding "
             "box")
        {
            geom.emplace_back(20 * volSize, volSize, pi_t / 2, volumeDescriptor, sinoDescriptor,
                              0.0, 0.0, -volSize);

            TestType op(volumeDescriptor, sinoDescriptor, geom);

            THEN("Result of forward projection is zero")
            {
                op.apply(volume, sino);
                DataContainer<data_t> zero(sinoDescriptor);
                zero = 0;
                REQUIRE(isApprox(sino, zero, epsilon));

                AND_THEN("Result of backprojection is zero")
                {
                    op.applyAdjoint(sino, volume);
                    DataContainer<data_t> zero(volumeDescriptor);
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
        DataContainer<data_t> volume(volumeDescriptor);
        DataContainer<data_t> sino(sinoDescriptor);
        volume = 1;
        sino = 1;
        std::vector<Geometry> geom;

        const index_t numCases = 9;
        std::array<real_t, numCases> alpha = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        std::array<real_t, numCases> beta = {0.0, 0.0,      0.0,      0.0,     0.0,
                                             0.0, pi_t / 2, pi_t / 2, pi_t / 2};
        std::array<real_t, numCases> gamma = {0.0,      0.0,      0.0,      pi_t / 2, pi_t / 2,
                                              pi_t / 2, pi_t / 2, pi_t / 2, pi_t / 2};
        std::array<real_t, numCases> offsetx = {volSize, 0.0,     volSize, 0.0,    0.0,
                                                0.0,     volSize, 0.0,     volSize};
        std::array<real_t, numCases> offsety = {0.0,     volSize, volSize, volSize, 0.0,
                                                volSize, 0.0,     0.0,     0.0};
        std::array<real_t, numCases> offsetz = {0.0,     0.0, 0.0,     0.0,    volSize,
                                                volSize, 0.0, volSize, volSize};
        std::array<std::string, numCases> neg = {"x",       "y", "x and y", "y",      "z",
                                                 "y and z", "x", "z",       "x and z"};
        std::array<std::string, numCases> ali = {"z", "z", "z", "x", "x", "x", "y", "y", "y"};

        for (std::size_t i = 0; i < numCases; i++) {
            WHEN("Tracing along a " + ali[i] + "-axis-aligned ray with negative " + neg[i]
                 + "-coodinate of origin")
            {
                geom.emplace_back(20 * volSize, volSize, volumeDescriptor, sinoDescriptor, gamma[i],
                                  beta[i], alpha[i], 0.0, 0.0, offsetx[i], offsety[i], offsetz[i]);

                TestType op(volumeDescriptor, sinoDescriptor, geom);

                THEN("Result of forward projection is zero")
                {
                    op.apply(volume, sino);
                    DataContainer<data_t> zero(sinoDescriptor);
                    zero = 0;
                    REQUIRE(isApprox(sino, zero, epsilon));

                    AND_THEN("Result of backprojection is zero")
                    {
                        op.applyAdjoint(sino, volume);
                        DataContainer<data_t> zero(volumeDescriptor);
                        zero = 0;
                        REQUIRE(isApprox(volume, zero, epsilon));
                    }
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("Scenario: Axis-aligned rays are present", "", SiddonsMethodCUDA<float>,
                   SiddonsMethodCUDA<double>, SiddonsMethod<float>, SiddonsMethod<double>)
{
    // Turn logger of
    Logger::setLevel(Logger::LogLevel::OFF);

    using data_t = decltype(return_data_t(std::declval<TestType>()));
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
        DataContainer<data_t> volume(volumeDescriptor);
        DataContainer<data_t> sino(sinoDescriptor);
        std::vector<Geometry> geom;

        const index_t numCases = 4;
        const std::array<real_t, numCases> angles = {0.0, pi_t / 2, pi_t, 3 * pi_t / 2};
        Eigen::Matrix<data_t, volSize * volSize, 1> backProj[2];
        backProj[1] << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

        backProj[0] << 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0;

        for (std::size_t i = 0; i < numCases; i++) {
            WHEN("An axis-aligned ray with an angle of " + std::to_string(angles[i])
                 + " radians passes through the center of a pixel")
            {
                geom.emplace_back(volSize * 20, volSize, angles[i], volumeDescriptor,
                                  sinoDescriptor);
                TestType op(volumeDescriptor, sinoDescriptor, geom);
                THEN("The result of projecting through a pixel is exactly the pixel value")
                {
                    for (index_t j = 0; j < volSize; j++) {
                        volume = 0;
                        if (i % 2 == 0)
                            volume(volSize / 2, j) = 1;
                        else
                            volume(j, volSize / 2) = 1;

                        op.apply(volume, sino);
                        REQUIRE(sino[0] == 1);
                    }

                    AND_THEN("The backprojection sets the values of all hit pixels to the detector "
                             "value")
                    {
                        op.applyAdjoint(sino, volume);
                        REQUIRE(isApprox(volume,
                                         DataContainer<data_t>(volumeDescriptor, backProj[i % 2])));
                    }
                }
            }
        }

        WHEN("A y-axis-aligned ray runs along the left voxel boundary")
        {
            geom.emplace_back(volSize * 2000, volSize, 0.0, volumeDescriptor, sinoDescriptor, 0.0,
                              -0.5);
            TestType op(volumeDescriptor, sinoDescriptor, geom);
            THEN("The result of projecting through a pixel is the value of the pixel with the "
                 "higher index")
            {
                for (index_t j = 0; j < volSize; j++) {
                    volume = 0;
                    volume(volSize / 2, j) = 1;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == Approx(1.0));
                }

                AND_THEN("The backprojection yields the exact adjoint")
                {
                    sino[0] = 1;
                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer<data_t>(volumeDescriptor, backProj[0])));
                }
            }
        }

        WHEN("A y-axis-aligned ray runs along the right volume boundary")
        {
            // For Siddons's values in the range [0,boxMax) are considered, a ray running along
            // boxMax should be ignored
            geom.emplace_back(volSize * 2000, volSize, 0.0, volumeDescriptor, sinoDescriptor, 0.0,
                              (volSize * 0.5));
            TestType op(volumeDescriptor, sinoDescriptor, geom);

            THEN("The result of projecting is zero")
            {
                volume = 0;
                op.apply(volume, sino);
                REQUIRE(sino[0] == 0.0);

                AND_THEN("The result of backprojection is also zero")
                {
                    sino[0] = 1;

                    op.applyAdjoint(sino, volume);
                    DataContainer<data_t> zero(volumeDescriptor);
                    zero = 0;
                    REQUIRE(volume == zero);
                }
            }
        }

        backProj[0] << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0;

        WHEN("A y-axis-aligned ray runs along the left volume boundary")
        {
            geom.emplace_back(volSize * 2000, volSize, 0.0, volumeDescriptor, sinoDescriptor, 0.0,
                              -volSize / 2.0);
            TestType op(volumeDescriptor, sinoDescriptor, geom);
            THEN("The result of projecting through a pixel is exactly the pixel's value")
            {
                for (index_t j = 0; j < volSize; j++) {
                    volume = 0;
                    volume(0, j) = 1;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == 1);
                }

                AND_THEN("The backprojection yields the exact adjoint")
                {
                    sino[0] = 1;
                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer<data_t>(volumeDescriptor, backProj[0])));
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
        DataContainer<data_t> volume(volumeDescriptor);
        DataContainer<data_t> sino(sinoDescriptor);
        std::vector<Geometry> geom;

        const index_t numCases = 6;
        std::array<real_t, numCases> beta = {0.0, 0.0, 0.0, 0.0, pi_t / 2, 3 * pi_t / 2};
        std::array<real_t, numCases> gamma = {0.0,          pi_t,     pi_t / 2,
                                              3 * pi_t / 2, pi_t / 2, 3 * pi_t / 2};
        std::array<std::string, numCases> al = {"z", "-z", "x", "-x", "y", "-y"};

        Eigen::Matrix<data_t, volSize * volSize * volSize, 1> backProj[numCases];

        backProj[2] << 0, 0, 0, 0, 0, 0, 0, 0, 0,

            0, 1, 0, 0, 1, 0, 0, 1, 0,

            0, 0, 0, 0, 0, 0, 0, 0, 0;

        backProj[1] << 0, 0, 0, 0, 0, 0, 0, 0, 0,

            0, 0, 0, 1, 1, 1, 0, 0, 0,

            0, 0, 0, 0, 0, 0, 0, 0, 0;

        backProj[0] << 0, 0, 0, 0, 1, 0, 0, 0, 0,

            0, 0, 0, 0, 1, 0, 0, 0, 0,

            0, 0, 0, 0, 1, 0, 0, 0, 0;

        for (std::size_t i = 0; i < numCases; i++) {
            WHEN("A " + al[i] + "-axis-aligned ray passes through the center of a pixel")
            {
                geom.emplace_back(volSize * 20, volSize, volumeDescriptor, sinoDescriptor, gamma[i],
                                  beta[i]);
                TestType op(volumeDescriptor, sinoDescriptor, geom);
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
                        REQUIRE(sino[0] == 1);
                    }

                    AND_THEN("The backprojection sets the values of all hit voxels to the detector "
                             "value")
                    {
                        op.applyAdjoint(sino, volume);
                        REQUIRE(isApprox(volume,
                                         DataContainer<data_t>(volumeDescriptor, backProj[i / 2])));
                    }
                }
            }
        }

        std::array<real_t, numCases> offsetx;
        std::array<real_t, numCases> offsety;

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
        al[2] = "bottom left border";
        al[3] = "right border";
        al[4] = "top border";
        al[5] = "top right edge";

        for (std::size_t i = 0; i < numCases / 2; i++) {
            WHEN("A z-axis-aligned ray runs along the " + al[i] + " of the volume")
            {
                // x-ray source must be very far from the volume center to make testing of the op
                // backprojection simpler
                geom.emplace_back(volSize * 2000, volSize, volumeDescriptor, sinoDescriptor, 0.0,
                                  0.0, 0.0, 0.0, 0.0, -offsetx[i], -offsety[i]);
                TestType op(volumeDescriptor, sinoDescriptor, geom);
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
                            case 2:
                                volume(0, 0, j) = 1;
                                break;
                            default:
                                break;
                        }

                        op.apply(volume, sino);
                        REQUIRE(sino[0] == 1);
                    }

                    AND_THEN("The backprojection yields the exact adjoints")
                    {
                        sino[0] = 1;
                        op.applyAdjoint(sino, volume);

                        REQUIRE(
                            isApprox(volume, DataContainer<data_t>(volumeDescriptor, backProj[i])));
                    }
                }
            }
        }

        for (std::size_t i = numCases / 2; i < numCases; i++) {
            WHEN("A z-axis-aligned ray runs along the " + al[i] + " of the volume")
            {
                // x-ray source must be very far from the volume center to make testing of the op
                // backprojection simpler
                geom.emplace_back(volSize * 2000, volSize, volumeDescriptor, sinoDescriptor, 0.0,
                                  0.0, 0.0, 0.0, 0.0, -offsetx[i], -offsety[i]);
                TestType op(volumeDescriptor, sinoDescriptor, geom);
                THEN("The result of projecting is zero")
                {
                    volume = 1;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == 0);

                    AND_THEN("The result of backprojection is also zero")
                    {
                        sino[0] = 1;
                        op.applyAdjoint(sino, volume);

                        DataContainer<data_t> zero(volumeDescriptor);
                        zero = 0;
                        REQUIRE(volume == zero);
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
        DataDescriptor volumeDescriptor(volumeDims);
        DataDescriptor sinoDescriptor(sinoDims);
        DataContainer<data_t> volume(volumeDescriptor);
        DataContainer<data_t> sino(sinoDescriptor);
        std::vector<Geometry> geom;

        WHEN("Both x- and y-axis-aligned rays are present")
        {
            geom.emplace_back(20 * volSize, volSize, 0, volumeDescriptor, sinoDescriptor);
            geom.emplace_back(20 * volSize, volSize, 90 * pi_t / 180., volumeDescriptor,
                              sinoDescriptor);
            geom.emplace_back(20 * volSize, volSize, 180 * pi_t / 180., volumeDescriptor,
                              sinoDescriptor);
            geom.emplace_back(20 * volSize, volSize, 270 * pi_t / 180., volumeDescriptor,
                              sinoDescriptor);

            TestType op(volumeDescriptor, sinoDescriptor, geom);

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
                    REQUIRE(sino[i] == Approx(5.0));

                AND_THEN("Backprojection yields the exact adjoint")
                {
                    Eigen::Matrix<data_t, Eigen::Dynamic, 1> cmp(volSize * volSize);

                    cmp << 0, 0, 10, 0, 0, 0, 0, 10, 0, 0, 10, 10, 20, 10, 10, 0, 0, 10, 0, 0, 0, 0,
                        10, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer<data_t>(volumeDescriptor, cmp)));
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
        DataContainer<data_t> volume(volumeDescriptor);
        DataContainer<data_t> sino(sinoDescriptor);
        std::vector<Geometry> geom;

        WHEN("x-, y and z-axis-aligned rays are present")
        {
            real_t beta[numImgs] = {0.0, 0.0, 0.0, 0.0, pi_t / 2, 3 * pi_t / 2};
            real_t gamma[numImgs] = {0.0, pi_t, pi_t / 2, 3 * pi_t / 2, pi_t / 2, 3 * pi_t / 2};

            for (index_t i = 0; i < numImgs; i++)
                geom.emplace_back(volSize * 20, volSize, volumeDescriptor, sinoDescriptor, gamma[i],
                                  beta[i]);

            TestType op(volumeDescriptor, sinoDescriptor, geom);

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
                    REQUIRE(sino[i] == Approx(3.0));

                AND_THEN("Backprojection yields the exact adjoint")
                {
                    Eigen::Matrix<data_t, Eigen::Dynamic, 1> cmp(volSize * volSize * volSize);

                    cmp << 0, 0, 0, 0, 6, 0, 0, 0, 0,

                        0, 6, 0, 6, 18, 6, 0, 6, 0,

                        0, 0, 0, 0, 6, 0, 0, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer<data_t>(volumeDescriptor, cmp)));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("Scenario: Projection under an angle", "", SiddonsMethodCUDA<float>,
                   SiddonsMethodCUDA<double>, SiddonsMethod<float>, SiddonsMethod<double>)
{
    // Turn logger of
    Logger::setLevel(Logger::LogLevel::OFF);

    using data_t = decltype(return_data_t(std::declval<TestType>()));

    real_t sqrt3r = std::sqrt(static_cast<real_t>(3));
    data_t sqrt3d = std::sqrt(static_cast<data_t>(3));

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
        DataContainer<data_t> volume(volumeDescriptor);
        DataContainer<data_t> sino(sinoDescriptor);
        std::vector<Geometry> geom;

        WHEN("Projecting under an angle of 30 degrees and ray goes through center of volume")
        {
            // In this case the ray enters and exits the volume through the borders along the main
            // direction
            geom.emplace_back(volSize * 20, volSize, -pi_t / 6, volumeDescriptor, sinoDescriptor);
            TestType op(volumeDescriptor, sinoDescriptor, geom);

            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                volume(3, 0) = 0;
                volume(2, 0) = 0;
                volume(2, 1) = 0;

                volume(1, 2) = 0;
                volume(1, 3) = 0;
                volume(0, 3) = 0;
                volume(2, 2) = 0;

                op.apply(volume, sino);
                DataContainer<data_t> zero(sinoDescriptor);
                zero = 0;
                REQUIRE(zero[0] == Approx(0).epsilon(epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    volume(3, 0) = 1;
                    volume(2, 0) = 2;
                    volume(2, 1) = 3;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == Approx(2 * sqrt3d + 2));

                    // on the other side of the center
                    volume = 0;
                    volume(1, 2) = 3;
                    volume(1, 3) = 2;
                    volume(0, 3) = 1;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == Approx(2 * sqrt3d + 2));

                    sino[0] = 1;

                    Eigen::Matrix<data_t, Eigen::Dynamic, 1> expected(volSize * volSize);
                    expected << 0, 0, 2 - 2 / sqrt3d, 4 / sqrt3d - 2, 0, 0, 2 / sqrt3d, 0, 0,
                        2 / sqrt3d, 0, 0, 4 / sqrt3d - 2, 2 - 2 / sqrt3d, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer<data_t>(volumeDescriptor, expected),
                                     epsilon));
                }
            }
        }

        WHEN("Projecting under an angle of 30 degrees and ray enters through the right border")
        {
            // In this case the ray exits through a border along the main ray direction, but enters
            // through a border not along the main direction
            geom.emplace_back(volSize * 20, volSize, -pi_t / 6, volumeDescriptor, sinoDescriptor,
                              0.0, sqrt3r);
            TestType op(volumeDescriptor, sinoDescriptor, geom);

            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                volume(3, 1) = 0;
                volume(3, 2) = 0;
                volume(3, 3) = 0;
                volume(2, 3) = 0;

                op.apply(volume, sino);
                DataContainer<data_t> zero(sinoDescriptor);
                zero = 0;
                REQUIRE(isApprox(sino, zero, epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    volume(3, 1) = 4;
                    volume(3, 2) = 3;
                    volume(3, 3) = 2;
                    volume(2, 3) = 1;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == Approx(14 - 4 * sqrt3d));

                    sino[0] = 1;

                    Eigen::Matrix<data_t, Eigen::Dynamic, 1> expected(volSize * volSize);
                    expected << 0, 0, 0, 0, 0, 0, 0, 4 - 2 * sqrt3d, 0, 0, 0, 2 / sqrt3d, 0, 0,
                        2 - 2 / sqrt3d, 4 / sqrt3d - 2;

                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer<data_t>(volumeDescriptor, expected),
                                     epsilon));
                }
            }
        }

        WHEN("Projecting under an angle of 30 degrees and ray exits through the left border")
        {
            // In this case the ray enters through a border along the main ray direction, but exits
            // through a border not along the main direction
            geom.emplace_back(volSize * 20, volSize, -pi_t / 6, volumeDescriptor, sinoDescriptor,
                              0.0, -sqrt3r);
            TestType op(volumeDescriptor, sinoDescriptor, geom);

            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                volume(0, 0) = 0;
                volume(1, 0) = 0;
                volume(0, 1) = 0;
                volume(0, 2) = 0;

                op.apply(volume, sino);
                DataContainer<data_t> zero(sinoDescriptor);
                zero = 0;
                REQUIRE(isApprox(sino, zero, epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    volume(1, 0) = 1;
                    volume(0, 0) = 2;
                    volume(0, 1) = 3;
                    volume(0, 2) = 4;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == Approx(14 - 4 * sqrt3d));

                    sino[0] = 1;

                    Eigen::Matrix<data_t, Eigen::Dynamic, 1> expected(volSize * volSize);
                    expected << 4 / sqrt3d - 2, 2 - 2 / sqrt3d, 0, 0, 2 / sqrt3d, 0, 0, 0,
                        4 - 2 * sqrt3d, 0, 0, 0, 0, 0, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer<data_t>(volumeDescriptor, expected),
                                     epsilon));
                }
            }
        }

        WHEN("Projecting under an angle of 30 degrees and ray only intersects a single pixel")
        {
            geom.emplace_back(volSize * 20, volSize, -pi_t / 6, volumeDescriptor, sinoDescriptor,
                              0.0, -2 - sqrt3r / 2);
            TestType op(volumeDescriptor, sinoDescriptor, geom);

            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                volume(0, 0) = 0;

                op.apply(volume, sino);
                DataContainer<data_t> zero(sinoDescriptor);
                zero = 0;
                REQUIRE(isApprox(sino, zero, epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    volume(0, 0) = 1;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == Approx(1 / sqrt3d));

                    sino[0] = 1;

                    Eigen::Matrix<data_t, Eigen::Dynamic, 1> expected(volSize * volSize);
                    expected << 1 / sqrt3d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer<data_t>(volumeDescriptor, expected),
                                     epsilon));
                }
            }
        }

        WHEN("Projecting under an angle of 120 degrees and ray goes through center of volume")
        {
            // In this case the ray enters and exits the volume through the borders along the main
            // direction
            geom.emplace_back(volSize * 20, volSize, -2 * pi_t / 3, volumeDescriptor,
                              sinoDescriptor);
            TestType op(volumeDescriptor, sinoDescriptor, geom);

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
                DataContainer<data_t> zero(sinoDescriptor);
                zero = 0;
                REQUIRE(zero[0] == Approx(0).epsilon(epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    volume(0, 0) = 1;
                    volume(0, 1) = 2;
                    volume(1, 1) = 3;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == Approx(2 * sqrt3d + 2));

                    // on the other side of the center
                    volume = 0;
                    volume(2, 2) = 3;
                    volume(3, 2) = 2;
                    volume(3, 3) = 1;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == Approx(2 * sqrt3d + 2));

                    sino[0] = 1;

                    Eigen::Matrix<data_t, Eigen::Dynamic, 1> expected(volSize * volSize);

                    expected << 4 / sqrt3d - 2, 0, 0, 0, 2 - 2 / sqrt3d, 2 / sqrt3d, 0, 0, 0, 0,
                        2 / sqrt3d, 2 - 2 / sqrt3d, 0, 0, 0, 4 / sqrt3d - 2;

                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer<data_t>(volumeDescriptor, expected),
                                     epsilon));
                }
            }
        }

        WHEN("Projecting under an angle of 120 degrees and ray enters through the top border")
        {
            // In this case the ray exits through a border along the main ray direction, but enters
            // through a border not along the main direction
            geom.emplace_back(volSize * 20, volSize, -2 * pi_t / 3, volumeDescriptor,
                              sinoDescriptor, 0.0, 0.0, sqrt3r);
            TestType op(volumeDescriptor, sinoDescriptor, geom);

            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                volume(0, 2) = 0;
                volume(0, 3) = 0;
                volume(1, 3) = 0;
                volume(2, 3) = 0;

                op.apply(volume, sino);
                DataContainer<data_t> zero(sinoDescriptor);
                zero = 0;
                REQUIRE(zero[0] == Approx(0).epsilon(epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    volume(0, 2) = 1;
                    volume(0, 3) = 2;
                    volume(1, 3) = 3;
                    volume(2, 3) = 4;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == Approx(14 - 4 * sqrt3d));

                    sino[0] = 1;

                    Eigen::Matrix<data_t, Eigen::Dynamic, 1> expected(volSize * volSize);

                    expected << 0, 0, 0, 0, 0, 0, 0, 0, 2 - 2 / sqrt3d, 0, 0, 0, 4 / sqrt3d - 2,
                        2 / sqrt3d, 4 - 2 * sqrt3d, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer<data_t>(volumeDescriptor, expected),
                                     epsilon));
                }
            }
        }

        WHEN("Projecting under an angle of 120 degrees and ray exits through the bottom border")
        {
            // In this case the ray enters through a border along the main ray direction, but exits
            // through a border not along the main direction
            geom.emplace_back(volSize * 20, volSize, -2 * pi_t / 3, volumeDescriptor,
                              sinoDescriptor, 0.0, 0.0, -sqrt3r);
            TestType op(volumeDescriptor, sinoDescriptor, geom);

            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                volume(1, 0) = 0;
                volume(2, 0) = 0;
                volume(3, 0) = 0;
                volume(3, 1) = 0;

                op.apply(volume, sino);
                DataContainer<data_t> zero(sinoDescriptor);
                zero = 0;
                REQUIRE(zero[0] == Approx(0).epsilon(epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    volume(1, 0) = 4;
                    volume(2, 0) = 3;
                    volume(3, 0) = 2;
                    volume(3, 1) = 1;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == Approx(14 - 4 * sqrt3d));

                    sino[0] = 1;

                    Eigen::Matrix<data_t, Eigen::Dynamic, 1> expected(volSize * volSize);

                    expected << 0, 4 - 2 * sqrt3d, 2 / sqrt3d, 4 / sqrt3d - 2, 0, 0, 0,
                        2 - 2 / sqrt3d, 0, 0, 0, 0, 0, 0, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer<data_t>(volumeDescriptor, expected),
                                     epsilon));
                }
            }
        }

        WHEN("Projecting under an angle of 120 degrees and ray only intersects a single pixel")
        {
            // This is a special case that is handled separately in both forward and backprojection
            geom.emplace_back(volSize * 20, volSize, -2 * pi_t / 3, volumeDescriptor,
                              sinoDescriptor, 0.0, 0.0, -2 - sqrt3r / 2);
            TestType op(volumeDescriptor, sinoDescriptor, geom);

            THEN("Ray intersects the correct pixels")
            {
                volume = 1;
                volume(3, 0) = 0;

                op.apply(volume, sino);
                DataContainer<data_t> zero(sinoDescriptor);
                zero = 0;
                REQUIRE(zero[0] == Approx(0).epsilon(epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    volume(3, 0) = 1;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == Approx(1 / sqrt3d).epsilon(0.005));

                    sino[0] = 1;

                    Eigen::Matrix<data_t, Eigen::Dynamic, 1> expected(volSize * volSize);
                    expected << 0, 0, 0, 1 / sqrt3d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer<data_t>(volumeDescriptor, expected),
                                     epsilon));
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
        DataContainer<data_t> volume(volumeDescriptor);
        DataContainer<data_t> sino(sinoDescriptor);
        std::vector<Geometry> geom;

        Eigen::Matrix<data_t, volSize * volSize * volSize, 1> backProj;

        WHEN("A ray with an angle of 30 degrees goes through the center of the volume")
        {
            // In this case the ray enters and exits the volume along the main direction
            geom.emplace_back(volSize * 20, volSize, volumeDescriptor, sinoDescriptor, pi_t / 6);
            TestType op(volumeDescriptor, sinoDescriptor, geom);

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
                    REQUIRE(sino[0] == Approx(3 * sqrt3d - 1).epsilon(epsilon));

                    sino[0] = 1;
                    backProj << 0, 0, 0, 0, 1 - 1 / sqrt3d, sqrt3d - 1, 0, 0, 0,

                        0, 0, 0, 0, 2 / sqrt3d, 0, 0, 0, 0,

                        0, 0, 0, sqrt3d - 1, 1 - 1 / sqrt3d, 0, 0, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer<data_t>(volumeDescriptor, backProj),
                                     epsilon));
                }
            }
        }

        WHEN("A ray with an angle of 30 degrees enters through the right border")
        {
            // In this case the ray enters through a border orthogonal to a non-main direction
            geom.emplace_back(volSize * 20, volSize, volumeDescriptor, sinoDescriptor, pi_t / 6,
                              0.0, 0.0, 0.0, 0.0, 1);
            TestType op(volumeDescriptor, sinoDescriptor, geom);

            THEN("The ray intersects the correct voxels")
            {
                volume = 1;
                volume(2, 1, 1) = 0;
                volume(2, 1, 0) = 0;
                volume(2, 1, 2) = 0;
                volume(1, 1, 2) = 0;

                op.apply(volume, sino);
                REQUIRE(sino[0] == Approx(0).epsilon(epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    volume(2, 1, 0) = 4;
                    volume(1, 1, 2) = 3;
                    volume(2, 1, 1) = 1;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == Approx(1 - 2 / sqrt3d + 3 * sqrt3d));

                    sino[0] = 1;
                    backProj << 0, 0, 0, 0, 0, 1 - 1 / sqrt3d, 0, 0, 0,

                        0, 0, 0, 0, 0, 2 / sqrt3d, 0, 0, 0,

                        0, 0, 0, 0, sqrt3d - 1, 1 - 1 / sqrt3d, 0, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer<data_t>(volumeDescriptor, backProj),
                                     epsilon));
                }
            }
        }

        WHEN("A ray with an angle of 30 degrees exits through the left border")
        {
            // In this case the ray exit through a border orthogonal to a non-main direction
            geom.emplace_back(volSize * 20, volSize, volumeDescriptor, sinoDescriptor, pi_t / 6,
                              0.0, 0.0, 0.0, 0.0, -1);
            TestType op(volumeDescriptor, sinoDescriptor, geom);

            THEN("The ray intersects the correct voxels")
            {
                volume = 1;
                volume(0, 1, 0) = 0;
                volume(1, 1, 0) = 0;
                volume(0, 1, 1) = 0;
                volume(0, 1, 2) = 0;

                op.apply(volume, sino);
                REQUIRE(sino[0] == Approx(0).epsilon(epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    volume(0, 1, 2) = 4;
                    volume(1, 1, 0) = 3;
                    volume(0, 1, 1) = 1;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == Approx(3 * sqrt3d + 1 - 2 / sqrt3d).epsilon(epsilon));

                    sino[0] = 1;
                    backProj << 0, 0, 0, 1 - 1 / sqrt3d, sqrt3d - 1, 0, 0, 0, 0,

                        0, 0, 0, 2 / sqrt3d, 0, 0, 0, 0, 0,

                        0, 0, 0, 1 - 1 / sqrt3d, 0, 0, 0, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer<data_t>(volumeDescriptor, backProj),
                                     epsilon));
                }
            }
        }

        WHEN("A ray with an angle of 30 degrees only intersects a single voxel")
        {
            // special case - no interior voxels, entry and exit voxels are the same
            geom.emplace_back(volSize * 20, volSize, volumeDescriptor, sinoDescriptor, pi_t / 6,
                              0.0, 0.0, 0.0, 0.0, -2);
            TestType op(volumeDescriptor, sinoDescriptor, geom);

            THEN("The ray intersects the correct voxels")
            {
                volume = 1;
                volume(0, 1, 0) = 0;

                op.apply(volume, sino);
                REQUIRE(sino[0] == Approx(0).epsilon(epsilon));

                AND_THEN("The correct weighting is applied")
                {
                    volume(0, 1, 0) = 1;

                    op.apply(volume, sino);
                    REQUIRE(sino[0] == Approx(sqrt3d - 1).epsilon(epsilon));

                    sino[0] = 1;
                    backProj << 0, 0, 0, sqrt3d - 1, 0, 0, 0, 0, 0,

                        0, 0, 0, 0, 0, 0, 0, 0, 0,

                        0, 0, 0, 0, 0, 0, 0, 0, 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE(isApprox(volume, DataContainer<data_t>(volumeDescriptor, backProj),
                                     epsilon));
                }
            }
        }
    }
}
