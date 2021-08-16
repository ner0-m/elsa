#include "doctest/doctest.h"

#include "SiddonsMethodCUDA.h"
#include "SiddonsMethod.h"
#include "Geometry.h"
#include "testHelpers.h"
#include "Logger.h"
#include "VolumeDescriptor.h"
#include "PlanarDetectorDescriptor.h"
#include "testHelpers.h"

#include <array>

using namespace elsa;
using namespace elsa::geometry;
using namespace doctest;

// TODO(dfrank): remove this and replace with checkApproxEq
using doctest::Approx;

/*
 * this function declaration can be used in conjunction with decltype to deduce the
 * template parameter of a templated class at compile time
 *
 * the template parameter must be a typename
 */
template <template <typename> typename T, typename data_t>
constexpr data_t return_data_t(const T<data_t>&);

TEST_CASE_TEMPLATE("SiddonsMethodCUDA: Calls to functions of super class", TestType,
                   SiddonsMethodCUDA<float>, SiddonsMethodCUDA<double>, SiddonsMethod<float>,
                   SiddonsMethod<double>)
{
    // Turn logger of
    Logger::setLevel(Logger::LogLevel::OFF);

    using data_t = decltype(return_data_t(std::declval<TestType>()));
    GIVEN("A projector")
    {
        IndexVector_t volumeDims(2), sinoDims(2);
        const index_t volSize = 10;
        const index_t detectorSize = 10;
        const index_t numImgs = 10;
        volumeDims << volSize, volSize;
        sinoDims << detectorSize, numImgs;
        VolumeDescriptor volumeDescriptor(volumeDims);
        DataContainer<data_t> volume(volumeDescriptor);
        volume = 1;

        auto stc = SourceToCenterOfRotation{20 * volSize};
        auto ctr = CenterOfRotationToDetector{volSize};

        std::vector<Geometry> geom;
        for (index_t i = 0; i < numImgs; i++) {
            real_t angle = static_cast<real_t>(i) * 2 * pi_t / 10;
            geom.emplace_back(stc, ctr, Radian{angle}, VolumeData2D{Size2D{volumeDims}},
                              SinogramData2D{Size2D{sinoDims}});
        }

        PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
        DataContainer<data_t> sino(sinoDescriptor);
        sino = 0;

        TestType op(volumeDescriptor, sinoDescriptor);

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
                REQUIRE_UNARY(isApprox<data_t>(sino, sinoClone, epsilon));

                op.applyAdjoint(sino, volume);
                opClone->applyAdjoint(sino, volumeClone);
                REQUIRE_UNARY(isApprox<data_t>(volume, volumeClone, epsilon));
            }
        }
    }
}

TEST_CASE_TEMPLATE("SiddonsMethodCUDA: 2D setup with a single ray", TestType,
                   SiddonsMethodCUDA<float>, SiddonsMethodCUDA<double>, SiddonsMethod<float>,
                   SiddonsMethod<double>)
{
    // Turn logger of
    Logger::setLevel(Logger::LogLevel::OFF);

    using data_t = decltype(return_data_t(std::declval<TestType>()));

    GIVEN("A 5x5 volume and detector of size 1, with 1 angle")
    {
        // Domain setup
        const index_t volSize = 5;

        IndexVector_t volumeDims(2);
        volumeDims << volSize, volSize;

        VolumeDescriptor volumeDescriptor(volumeDims);
        DataContainer<data_t> volume(volumeDescriptor);

        // range setup
        const index_t detectorSize = 1;
        const index_t numImgs = 1;

        IndexVector_t sinoDims(2);
        sinoDims << detectorSize, numImgs;

        // Setup geometry
        auto stc = SourceToCenterOfRotation{20 * volSize};
        auto ctr = CenterOfRotationToDetector{volSize};
        auto volData = VolumeData2D{Size2D{volumeDims}};
        auto sinoData = SinogramData2D{Size2D{sinoDims}};

        std::vector<Geometry> geom;

        GIVEN("A basic geometry setup")
        {
            geom.emplace_back(stc, ctr, Radian{0}, std::move(volData), std::move(sinoData));

            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer<data_t> sino(sinoDescriptor);

            TestType op(volumeDescriptor, sinoDescriptor);

            WHEN("Sinogram conatainer is not zero initialized and we project through an empty "
                 "volume")
            {
                volume = 0;
                sino = 1;

                THEN("Result is zero")
                {
                    op.apply(volume, sino);
                    DataContainer<data_t> zero(sinoDescriptor);
                    zero = 0;
                    REQUIRE_UNARY(isApprox(sino, zero, epsilon));
                }
            }

            WHEN("Volume container is not zero initialized and we backproject from an empty "
                 "sinogram")
            {
                sino = 0;
                volume = 1;

                THEN("Result is zero")
                {
                    op.applyAdjoint(sino, volume);
                    DataContainer<data_t> zero(volumeDescriptor);
                    zero = 0;
                    REQUIRE_UNARY(isApprox(volume, zero, epsilon));
                }
            }
        }

        GIVEN("Scenario: Rays not intersecting the bounding box are present")
        {
            WHEN("Tracing along a y-axis-aligned ray with a negative x-coordinate of origin")
            {
                geom.emplace_back(stc, ctr, Radian{0}, std::move(volData), std::move(sinoData),
                                  PrincipalPointOffset{}, RotationOffset2D{volSize, 0});

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer<data_t> sino(sinoDescriptor);

                TestType op(volumeDescriptor, sinoDescriptor);

                THEN("Result of forward projection is zero")
                {
                    DataContainer<data_t> zero(sinoDescriptor);
                    zero = 0;

                    op.apply(volume, sino);
                    REQUIRE_UNARY(isApprox(sino, zero));

                    AND_THEN("Result of backprojection is zero")
                    {
                        DataContainer<data_t> zero(volumeDescriptor);
                        zero = 0;

                        op.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(isApprox(volume, zero));
                    }
                }
            }

            WHEN("Tracing along a y-axis-aligned ray with a x-coordinate of origin beyond the "
                 "bounding box")
            {
                geom.emplace_back(stc, ctr, Radian{0}, std::move(volData), std::move(sinoData),
                                  PrincipalPointOffset{}, RotationOffset2D{-volSize, 0});

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer<data_t> sino(sinoDescriptor);

                TestType op(volumeDescriptor, sinoDescriptor);

                THEN("Result of forward projection is zero")
                {
                    DataContainer<data_t> zero(sinoDescriptor);
                    zero = 0;

                    op.apply(volume, sino);
                    REQUIRE_UNARY(isApprox(sino, zero));

                    AND_THEN("Result of backprojection is zero")
                    {
                        DataContainer<data_t> zero(volumeDescriptor);
                        zero = 0;

                        op.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(isApprox(volume, zero));
                    }
                }
            }

            WHEN("Tracing along a x-axis-aligned ray with a negative y-coordinate of origin")
            {
                geom.emplace_back(stc, ctr, Radian{pi_t / 2}, std::move(volData),
                                  std::move(sinoData), PrincipalPointOffset{},
                                  RotationOffset2D{0, volSize});

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer<data_t> sino(sinoDescriptor);

                TestType op(volumeDescriptor, sinoDescriptor);

                THEN("Result of forward projection is zero")
                {
                    DataContainer<data_t> zero(sinoDescriptor);
                    zero = 0;

                    op.apply(volume, sino);
                    REQUIRE_UNARY(isApprox(sino, zero));

                    AND_THEN("Result of backprojection is zero")
                    {
                        DataContainer<data_t> zero(volumeDescriptor);
                        zero = 0;

                        op.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(isApprox(volume, zero));
                    }
                }
            }

            WHEN("Tracing along a x-axis-aligned ray with a y-coordinate of origin beyond the "
                 "bounding box")
            {
                geom.emplace_back(stc, ctr, Radian{pi_t / 2}, std::move(volData),
                                  std::move(sinoData), PrincipalPointOffset{},
                                  RotationOffset2D{0, -volSize});

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer<data_t> sino(sinoDescriptor);

                TestType op(volumeDescriptor, sinoDescriptor);

                THEN("Result of forward projection is zero")
                {
                    op.apply(volume, sino);
                    DataContainer<data_t> zero(sinoDescriptor);
                    zero = 0;

                    REQUIRE_UNARY(isApprox(sino, zero));

                    AND_THEN("Result of backprojection is zero")
                    {
                        op.applyAdjoint(sino, volume);

                        DataContainer<data_t> zero(volumeDescriptor);
                        zero = 0;

                        REQUIRE_UNARY(isApprox(volume, zero));
                    }
                }
            }
        }

        // Expected results
        Eigen::Matrix<data_t, Eigen::Dynamic, 1> backProjections[2];
        backProjections[0].resize(volSize * volSize);
        backProjections[1].resize(volSize * volSize);

        constexpr index_t numCases = 4;
        const std::array<real_t, numCases> angles = {0., 90., 180., 270.};

        // clang-format off
        backProjections[0] << 0, 0, 1, 0, 0,
                              0, 0, 1, 0, 0,
                              0, 0, 1, 0, 0,
                              0, 0, 1, 0, 0,
                              0, 0, 1, 0, 0;

        backProjections[1] << 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0,
                              1, 1, 1, 1, 1,
                              0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0;
        // clang-format on

        for (index_t i = 0; i < numCases; i++) {
            WHEN("An axis-aligned ray with a fixed angle passes through the center of a pixel")
            {
                INFO("An axis-aligned ray with an angle of ", angles[asUnsigned(i)],
                     " radians passes through the center of a pixel");

                geom.emplace_back(stc, ctr, Degree{angles[asUnsigned(i)]}, std::move(volData),
                                  std::move(sinoData));

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer<data_t> sino(sinoDescriptor);

                TestType op(volumeDescriptor, sinoDescriptor);

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
                            isApprox(volume, DataContainer<data_t>(volumeDescriptor,
                                                                   backProjections[i % 2])));
                    }
                }
            }
        }

        WHEN("A y-axis-aligned ray runs along the left voxel boundary")
        {
            geom.emplace_back(SourceToCenterOfRotation{volSize * 2000}, ctr, Radian{0},
                              std::move(volData), std::move(sinoData), PrincipalPointOffset{0},
                              RotationOffset2D{-0.5, 0});

            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer<data_t> sino(sinoDescriptor);

            TestType op(volumeDescriptor, sinoDescriptor);

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
                    REQUIRE_UNARY(isApprox(
                        volume, DataContainer<data_t>(volumeDescriptor, backProjections[0])));
                }
            }
        }

        WHEN("A y-axis-aligned ray runs along the right volume boundary")
        {
            // For Siddons's values in the range [0,boxMax) are considered, a ray running along
            // boxMax should be ignored
            geom.emplace_back(SourceToCenterOfRotation{volSize * 2000}, ctr, Radian{0},
                              std::move(volData), std::move(sinoData), PrincipalPointOffset{0},
                              RotationOffset2D{volSize * 0.5, 0});

            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer<data_t> sino(sinoDescriptor);

            TestType op(volumeDescriptor, sinoDescriptor);

            THEN("The result of projecting is zero")
            {
                volume = 0;
                op.apply(volume, sino);
                REQUIRE_EQ(sino[0], Approx(0));

                AND_THEN("The result of backprojection is also zero")
                {
                    sino[0] = 1;

                    op.applyAdjoint(sino, volume);
                    DataContainer<data_t> zero(volumeDescriptor);
                    zero = 0;
                    REQUIRE_UNARY(isApprox(volume, zero));
                }
            }
        }

        // clang-format off
        backProjections[0] << 1, 0, 0, 0, 0,
                              1, 0, 0, 0, 0,
                              1, 0, 0, 0, 0,
                              1, 0, 0, 0, 0,
                              1, 0, 0, 0, 0;
        // clang-format on

        WHEN("A y-axis-aligned ray runs along the left volume boundary")
        {
            geom.emplace_back(SourceToCenterOfRotation{volSize * 2000}, ctr, Radian{0},
                              std::move(volData), std::move(sinoData), PrincipalPointOffset{0},
                              RotationOffset2D{-volSize * 0.5, 0});

            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer<data_t> sino(sinoDescriptor);

            TestType op(volumeDescriptor, sinoDescriptor);

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
                    REQUIRE_UNARY(isApprox(
                        volume, DataContainer<data_t>(volumeDescriptor, backProjections[0])));
                }
            }
        }
    }
    GIVEN("a 4x4 Volume")
    {
        // Domain setup
        const index_t volSize = 4;

        IndexVector_t volumeDims(2);
        volumeDims << volSize, volSize;

        VolumeDescriptor volumeDescriptor(volumeDims);
        DataContainer<data_t> volume(volumeDescriptor);

        // range setup
        const index_t detectorSize = 1;
        const index_t numImgs = 1;

        IndexVector_t sinoDims(2);
        sinoDims << detectorSize, numImgs;

        // Setup geometry
        auto stc = SourceToCenterOfRotation{20 * volSize};
        auto ctr = CenterOfRotationToDetector{volSize};
        auto volData = VolumeData2D{Size2D{volumeDims}};
        auto sinoData = SinogramData2D{Size2D{sinoDims}};

        std::vector<Geometry> geom;

        real_t sqrt3r = std::sqrt(static_cast<real_t>(3));
        data_t sqrt3d = std::sqrt(static_cast<data_t>(3));

        GIVEN("An angle of -30 degrees")
        {
            auto angle = Degree{-30};
            WHEN("A ray goes through center of volume")
            {
                // In this case the ray enters and exits the volume through the borders along the
                // main direction
                geom.emplace_back(stc, ctr, angle, std::move(volData), std::move(sinoData));

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer<data_t> sino(sinoDescriptor);

                TestType op(volumeDescriptor, sinoDescriptor);

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

                    DataContainer<data_t> zero(sinoDescriptor);
                    zero = 0;

                    op.apply(volume, sino);
                    REQUIRE_EQ(zero[0], Approx(0).epsilon(epsilon));

                    AND_THEN("The correct weighting is applied")
                    {
                        volume(3, 0) = 1;
                        volume(2, 0) = 2;
                        volume(2, 1) = 3;

                        op.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx(2 * sqrt3d + 2));

                        // on the other side of the center
                        volume = 0;
                        volume(1, 2) = 3;
                        volume(1, 3) = 2;
                        volume(0, 3) = 1;

                        op.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx(2 * sqrt3d + 2));

                        sino[0] = 1;

                        Eigen::Matrix<data_t, Eigen::Dynamic, 1> expected(volSize * volSize);

                        // clang-format off
                        expected <<              0,              0, 2 - 2 / sqrt3d, 4 / sqrt3d - 2,
                                                 0,              0,     2 / sqrt3d,              0,
                                                 0,     2 / sqrt3d,              0,              0,
                                    4 / sqrt3d - 2, 2 - 2 / sqrt3d,              0,              0;
                        // clang-format on

                        op.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(isApprox(
                            volume, DataContainer<data_t>(volumeDescriptor, expected), epsilon));
                    }
                }
            }

            WHEN("A ray enters through the right border")
            {
                // In this case the ray exits through a border along the main ray direction, but
                // enters through a border not along the main direction First pixel should be
                // weighted differently
                geom.emplace_back(stc, ctr, angle, std::move(volData), std::move(sinoData),
                                  PrincipalPointOffset{0}, RotationOffset2D{sqrt3r, 0});

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer<data_t> sino(sinoDescriptor);

                TestType op(volumeDescriptor, sinoDescriptor);

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
                    REQUIRE_UNARY(isApprox(sino, zero, epsilon));

                    AND_THEN("The correct weighting is applied")
                    {
                        volume(3, 1) = 4;
                        volume(3, 2) = 3;
                        volume(3, 3) = 2;
                        volume(2, 3) = 1;

                        op.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx(14 - 4 * sqrt3d));

                        sino[0] = 1;

                        Eigen::Matrix<data_t, Eigen::Dynamic, 1> expected(volSize * volSize);

                        // clang-format off
                        expected << 0, 0,              0,              0,
                                    0, 0,              0, 4 - 2 * sqrt3d,
                                    0, 0,              0,     2 / sqrt3d,
                                    0, 0, 2 - 2 / sqrt3d, 4 / sqrt3d - 2;
                        // clang-format on

                        op.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(isApprox(
                            volume, DataContainer<data_t>(volumeDescriptor, expected), epsilon));
                    }
                }
            }

            WHEN("A ray exits through the left border")
            {
                // In this case the ray enters through a border along the main ray direction, but
                // exits through a border not along the main direction
                geom.emplace_back(stc, ctr, angle, std::move(volData), std::move(sinoData),
                                  PrincipalPointOffset{0}, RotationOffset2D{-sqrt3r, 0});

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer<data_t> sino(sinoDescriptor);

                TestType op(volumeDescriptor, sinoDescriptor);

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
                    REQUIRE_UNARY(isApprox(sino, zero, epsilon));

                    AND_THEN("The correct weighting is applied")
                    {
                        volume(1, 0) = 1;
                        volume(0, 0) = 2;
                        volume(0, 1) = 3;
                        volume(0, 2) = 4;

                        op.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx(14 - 4 * sqrt3d));

                        sino[0] = 1;

                        Eigen::Matrix<data_t, Eigen::Dynamic, 1> expected(volSize * volSize);

                        // clang-format off
                        expected << 4 / sqrt3d - 2, 2 - 2 / sqrt3d, 0, 0,
                                        2 / sqrt3d,              0, 0, 0,
                                    4 - 2 * sqrt3d,              0, 0, 0,
                                                 0,              0, 0, 0;
                        // clang-format on

                        op.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(isApprox(
                            volume, DataContainer<data_t>(volumeDescriptor, expected), epsilon));
                    }
                }
            }

            WHEN("A ray only intersects a single pixel")
            {
                geom.emplace_back(stc, ctr, angle, std::move(volData), std::move(sinoData),
                                  PrincipalPointOffset{0}, RotationOffset2D{-2 - sqrt3r / 2, 0});

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer<data_t> sino(sinoDescriptor);

                TestType op(volumeDescriptor, sinoDescriptor);

                THEN("Ray intersects the correct pixels")
                {
                    volume = 1;
                    volume(0, 0) = 0;

                    op.apply(volume, sino);
                    DataContainer<data_t> zero(sinoDescriptor);
                    zero = 0;
                    REQUIRE_UNARY(isApprox(sino, zero, epsilon));

                    AND_THEN("The correct weighting is applied")
                    {
                        volume(0, 0) = 1;

                        op.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx(1 / sqrt3d));

                        sino[0] = 1;

                        Eigen::Matrix<data_t, Eigen::Dynamic, 1> expected(volSize * volSize);
                        // clang-format off
                        expected << 1 / sqrt3d, 0, 0, 0,
                                             0, 0, 0, 0,
                                             0, 0, 0, 0,
                                             0, 0, 0, 0;
                        // clang-format on

                        op.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(isApprox(
                            volume, DataContainer<data_t>(volumeDescriptor, expected), epsilon));
                    }
                }
            }
        }

        GIVEN("An angle of 120 degrees")
        {
            auto angle = Degree{-120};

            WHEN("A ray goes through center of volume")
            {
                // In this case the ray enters and exits the volume through the borders along the
                // main direction
                geom.emplace_back(stc, ctr, angle, std::move(volData), std::move(sinoData));

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer<data_t> sino(sinoDescriptor);

                TestType op(volumeDescriptor, sinoDescriptor);

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
                    REQUIRE_EQ(zero[0], Approx(0).epsilon(epsilon));

                    AND_THEN("The correct weighting is applied")
                    {
                        volume(0, 0) = 1;
                        volume(0, 1) = 2;
                        volume(1, 1) = 3;

                        op.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx(2 * sqrt3d + 2));

                        // on the other side of the center
                        volume = 0;
                        volume(2, 2) = 3;
                        volume(3, 2) = 2;
                        volume(3, 3) = 1;

                        op.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx(2 * sqrt3d + 2));

                        sino[0] = 1;

                        Eigen::Matrix<data_t, Eigen::Dynamic, 1> expected(volSize * volSize);

                        // clang-format off
                        expected << 4 / sqrt3d - 2,          0,          0,              0,
                                    2 - 2 / sqrt3d, 2 / sqrt3d,          0,              0,
                                                 0,          0, 2 / sqrt3d, 2 - 2 / sqrt3d,
                                                 0,          0,          0, 4 / sqrt3d - 2;
                        // clang-format on

                        op.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(isApprox(
                            volume, DataContainer<data_t>(volumeDescriptor, expected), epsilon));
                    }
                }
            }

            WHEN("A ray enters through the top border")
            {
                // In this case the ray exits through a border along the main ray direction, but
                // enters through a border not along the main direction
                geom.emplace_back(stc, ctr, angle, std::move(volData), std::move(sinoData),
                                  PrincipalPointOffset{0}, RotationOffset2D{0, std::sqrt(3.f)});

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer<data_t> sino(sinoDescriptor);

                TestType op(volumeDescriptor, sinoDescriptor);

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
                    REQUIRE_EQ(zero[0], Approx(0).epsilon(epsilon));

                    AND_THEN("The correct weighting is applied")
                    {
                        volume(0, 2) = 1;
                        volume(0, 3) = 2;
                        volume(1, 3) = 3;
                        volume(2, 3) = 4;

                        op.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx(14 - 4 * sqrt3d));

                        sino[0] = 1;

                        Eigen::Matrix<data_t, Eigen::Dynamic, 1> expected(volSize * volSize);

                        // clang-format off
                        expected <<              0,              0,              0, 0,
                                                 0,              0,              0, 0,
                                    2 - 2 / sqrt3d,              0,              0, 0,
                                    4 / sqrt3d - 2,     2 / sqrt3d, 4 - 2 * sqrt3d, 0;
                        // clang-format on

                        op.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(isApprox(
                            volume, DataContainer<data_t>(volumeDescriptor, expected), epsilon));
                    }
                }
            }

            WHEN("A ray exits through the bottom border")

            {
                // In this case the ray enters through a border along the main ray direction, but
                // exits through a border not along the main direction
                geom.emplace_back(stc, ctr, angle, std::move(volData), std::move(sinoData),
                                  PrincipalPointOffset{0}, RotationOffset2D{0, -std::sqrt(3.f)});

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer<data_t> sino(sinoDescriptor);

                TestType op(volumeDescriptor, sinoDescriptor);

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
                    REQUIRE_EQ(zero[0], Approx(0).epsilon(epsilon));

                    AND_THEN("The correct weighting is applied")
                    {
                        volume(1, 0) = 4;
                        volume(2, 0) = 3;
                        volume(3, 0) = 2;
                        volume(3, 1) = 1;

                        op.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx(14 - 4 * sqrt3d));

                        sino[0] = 1;

                        Eigen::Matrix<data_t, Eigen::Dynamic, 1> expected(volSize * volSize);

                        // clang-format off
                        expected << 0, 4 - 2 * sqrt3d, 2 / sqrt3d, 4 / sqrt3d - 2, 0, 0, 0,
                            2 - 2 / sqrt3d, 0, 0, 0, 0, 0, 0, 0, 0;
                        // clang-format on

                        op.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(isApprox(
                            volume, DataContainer<data_t>(volumeDescriptor, expected), epsilon));
                    }
                }
            }

            WHEN("A ray only intersects a single pixel")
            {
                // This is a special case that is handled separately in both forward and
                // backprojection
                geom.emplace_back(stc, ctr, angle, std::move(volData), std::move(sinoData),
                                  PrincipalPointOffset{0},
                                  RotationOffset2D{0, -2 - std::sqrt(3.f) / 2});

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer<data_t> sino(sinoDescriptor);

                TestType op(volumeDescriptor, sinoDescriptor);

                THEN("Ray intersects the correct pixels")
                {
                    volume = 1;
                    volume(3, 0) = 0;

                    op.apply(volume, sino);
                    DataContainer<data_t> zero(sinoDescriptor);
                    zero = 0;
                    REQUIRE_EQ(zero[0], Approx(0).epsilon(epsilon));

                    AND_THEN("The correct weighting is applied")
                    {
                        volume(3, 0) = 1;

                        op.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx(1 / sqrt3d).epsilon(0.005));

                        sino[0] = 1;

                        Eigen::Matrix<data_t, Eigen::Dynamic, 1> expected(volSize * volSize);

                        // clang-format off
                        expected << 0, 0, 0, 1 / sqrt3d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                        // clang-format on

                        op.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(isApprox(
                            volume, DataContainer<data_t>(volumeDescriptor, expected), epsilon));
                    }
                }
            }
        }
    }
}

TEST_CASE_TEMPLATE("SiddonsMethodCUDA: 2D setup with a multiple rays", TestType,
                   SiddonsMethodCUDA<float>, SiddonsMethodCUDA<double>, SiddonsMethod<float>,
                   SiddonsMethod<double>)
{
    // Turn logger of
    Logger::setLevel(Logger::LogLevel::OFF);

    using data_t = decltype(return_data_t(std::declval<TestType>()));

    GIVEN("Given a 5x5 volume")
    {
        // Domain setup
        const index_t volSize = 5;

        IndexVector_t volumeDims(2);
        volumeDims << volSize, volSize;

        VolumeDescriptor volumeDescriptor(volumeDims);
        DataContainer<data_t> volume(volumeDescriptor);

        // range setup
        const index_t detectorSize = 1;
        const index_t numImgs = 4;

        IndexVector_t sinoDims(2);
        sinoDims << detectorSize, numImgs;

        // Setup geometry
        auto stc = SourceToCenterOfRotation{20 * volSize};
        auto ctr = CenterOfRotationToDetector{volSize};
        auto volData = VolumeData2D{Size2D{volumeDims}};
        auto sinoData = SinogramData2D{Size2D{sinoDims}};

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
            DataContainer<data_t> sino(sinoDescriptor);

            TestType op(volumeDescriptor, sinoDescriptor);

            THEN("Values are accumulated correctly along each ray's path")
            {
                volume = 0;

                // set only values along the rays' path to one to make sure interpolation is
                // dones correctly
                for (index_t i = 0; i < volSize; i++) {
                    volume(i, volSize / 2) = 1;
                    volume(volSize / 2, i) = 1;
                }

                op.apply(volume, sino);
                for (index_t i = 0; i < numImgs; i++)
                    REQUIRE_EQ(sino[i], Approx(5.0));

                AND_THEN("Backprojection yields the exact adjoint")
                {
                    Eigen::Matrix<data_t, Eigen::Dynamic, 1> cmp(volSize * volSize);

                    // clang-format off
                    cmp <<  0,  0, 10,  0,  0,
                            0,  0, 10,  0,  0,
                           10, 10, 20, 10, 10,
                            0,  0, 10,  0,  0,
                            0,  0, 10,  0,  0;
                    // clang-format on

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isApprox(volume, DataContainer<data_t>(volumeDescriptor, cmp)));
                }
            }
        }
    }
}

TEST_CASE_TEMPLATE("SiddonsMethodCUDA: 3D setup with a single ray", TestType,
                   SiddonsMethodCUDA<float>, SiddonsMethodCUDA<double>, SiddonsMethod<float>,
                   SiddonsMethod<double>)
{
    // Turn logger of
    Logger::setLevel(Logger::LogLevel::OFF);

    using data_t = decltype(return_data_t(std::declval<TestType>()));

    GIVEN("Given a 3x3x3 volume")
    {
        // Domain setup
        const index_t volSize = 3;

        IndexVector_t volumeDims(3);
        volumeDims << volSize, volSize, volSize;

        VolumeDescriptor volumeDescriptor(volumeDims);
        DataContainer<data_t> volume(volumeDescriptor);

        // range setup
        const index_t detectorSize = 1;
        const index_t numImgs = 1;

        IndexVector_t sinoDims(3);
        sinoDims << detectorSize, detectorSize, numImgs;

        // Setup geometry
        auto stc = SourceToCenterOfRotation{20 * volSize};
        auto ctr = CenterOfRotationToDetector{volSize};
        auto volData = VolumeData3D{Size3D{volumeDims}};
        auto sinoData = SinogramData3D{Size3D{sinoDims}};

        std::vector<Geometry> geom;

        GIVEN("A basic 3D setup")
        {
            geom.emplace_back(stc, ctr, std::move(volData), std::move(sinoData),
                              RotationAngles3D{Gamma{0}});

            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer<data_t> sino(sinoDescriptor);

            TestType op(volumeDescriptor, sinoDescriptor);

            WHEN("Sinogram conatainer is not zero initialized and we project through an empty "
                 "volume")
            {
                volume = 0;
                sino = 1;

                THEN("Result is zero")
                {
                    DataContainer<data_t> zero(sinoDescriptor);
                    zero = 0;

                    op.apply(volume, sino);
                    REQUIRE_UNARY(isApprox(sino, zero, epsilon));
                }
            }

            WHEN("Volume container is not zero initialized and we backproject from an empty "
                 "sinogram")
            {
                sino = 0;
                volume = 1;

                THEN("Result is zero")
                {
                    DataContainer<data_t> zero(volumeDescriptor);
                    zero = 0;

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isApprox(volume, zero, epsilon));
                }
            }
        }

        GIVEN("A rays along different axes")
        {
            const index_t numCases = 6;

            using RealArray = std::array<real_t, numCases>;
            using StrArray = std::array<std::string, numCases>;

            RealArray beta = {0.0, 0.0, 0.0, 0.0, pi_t / 2, 3 * pi_t / 2};
            RealArray gamma = {0.0, pi_t, pi_t / 2, 3 * pi_t / 2, pi_t / 2, 3 * pi_t / 2};
            StrArray al = {"z", "-z", "x", "-x", "y", "-y"};

            Eigen::Matrix<data_t, volSize * volSize * volSize, 1> backProj[numCases];

            // clang-format off
            backProj[0] << 0, 0, 0, 0, 1, 0, 0, 0, 0,
                           0, 0, 0, 0, 1, 0, 0, 0, 0,
                           0, 0, 0, 0, 1, 0, 0, 0, 0;

            backProj[1] << 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 1, 1, 1, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0;

            backProj[2] << 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 1, 0, 0, 1, 0, 0, 1, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0;
            // clang-format on

            for (index_t i = 0; i < numCases; i++) {
                WHEN("An axis-aligned ray passes through the center of a pixel")
                {
                    INFO("A ", al[asUnsigned(i)],
                         "-axis-aligned ray passes through the center of a pixel");

                    geom.emplace_back(
                        stc, ctr, std::move(volData), std::move(sinoData),
                        RotationAngles3D{Gamma{gamma[asUnsigned(i)]}, Beta{beta[asUnsigned(i)]}});

                    PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                    DataContainer<data_t> sino(sinoDescriptor);

                    TestType op(volumeDescriptor, sinoDescriptor);

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

                        AND_THEN(
                            "The backprojection sets the values of all hit voxels to the detector "
                            "value")
                        {
                            op.applyAdjoint(sino, volume);
                            REQUIRE_UNARY(isApprox(
                                volume, DataContainer<data_t>(volumeDescriptor, backProj[i / 2])));
                        }
                    }
                }
            }

            RealArray offsetx;
            RealArray offsety;

            offsetx[0] = volSize / 2.0;
            offsetx[1] = 0.0;
            offsetx[2] = (volSize / 2.0);
            offsetx[3] = -(volSize / 2.0);
            offsetx[4] = 0.0;
            offsetx[5] = -(volSize / 2.0);

            offsety[0] = 0.0;
            offsety[1] = volSize / 2.0;
            offsety[2] = (volSize / 2.0);
            offsety[3] = 0.0;
            offsety[4] = -(volSize / 2.0);
            offsety[5] = -(volSize / 2.0);

            // clang-format off
            backProj[0] << 0, 0, 0, 1, 0, 0, 0, 0, 0,
                           0, 0, 0, 1, 0, 0, 0, 0, 0,
                           0, 0, 0, 1, 0, 0, 0, 0, 0;

            backProj[1] << 0, 1, 0, 0, 0, 0, 0, 0, 0,
                           0, 1, 0, 0, 0, 0, 0, 0, 0,
                           0, 1, 0, 0, 0, 0, 0, 0, 0;

            backProj[2] << 1, 0, 0, 0, 0, 0, 0, 0, 0,
                           1, 0, 0, 0, 0, 0, 0, 0, 0,
                           1, 0, 0, 0, 0, 0, 0, 0, 0;
            // clang-format on

            al[0] = "left border";
            al[1] = "bottom border";
            al[2] = "bottom left border";
            al[3] = "right border";
            al[4] = "top border";
            al[5] = "top right edge";

            for (unsigned int i = 0; i < numCases / 2; i++) {
                WHEN("A z-axis-aligned ray runs along the corners and edges of the volume")
                {
                    INFO("A z-axis-aligned ray runs along the ", al[i], " of the volume");

                    // x-ray source must be very far from the volume center to make testing of the
                    // op backprojection simpler
                    geom.emplace_back(SourceToCenterOfRotation{volSize * 2000}, ctr,
                                      std::move(volData), std::move(sinoData),
                                      RotationAngles3D{Gamma{0}}, PrincipalPointOffset2D{0, 0},
                                      RotationOffset3D{-offsetx[i], -offsety[i], 0});

                    PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                    DataContainer<data_t> sino(sinoDescriptor);

                    TestType op(volumeDescriptor, sinoDescriptor);

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
                            REQUIRE_EQ(sino[0], Approx(1));
                        }

                        AND_THEN("The backprojection yields the exact adjoints")
                        {
                            sino[0] = 1;
                            op.applyAdjoint(sino, volume);

                            REQUIRE_UNARY(isApprox(
                                volume, DataContainer<data_t>(volumeDescriptor, backProj[i])));
                        }
                    }
                }
            }

            for (unsigned i = numCases / 2; i < numCases; i++) {
                WHEN("A z-axis-aligned ray runs along the corners and edges of the volume")
                {
                    INFO("A z-axis-aligned ray runs along the ", al[i], " of the volume");
                    // x-ray source must be very far from the volume center to make testing of the
                    // op backprojection simpler
                    geom.emplace_back(SourceToCenterOfRotation{volSize * 2000}, ctr,
                                      std::move(volData), std::move(sinoData),
                                      RotationAngles3D{Gamma{0}}, PrincipalPointOffset2D{0, 0},
                                      RotationOffset3D{-offsetx[i], -offsety[i], 0});

                    PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                    DataContainer<data_t> sino(sinoDescriptor);

                    TestType op(volumeDescriptor, sinoDescriptor);

                    THEN("The result of projecting is zero")
                    {
                        volume = 1;

                        op.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx(0));

                        AND_THEN("The result of backprojection is also zero")
                        {
                            sino[0] = 1;
                            op.applyAdjoint(sino, volume);

                            DataContainer<data_t> zero(volumeDescriptor);
                            zero = 0;
                            REQUIRE_UNARY(isCwiseApprox(volume, zero));
                        }
                    }
                }
            }
        }

        GIVEN("An angle of 30 degrees")
        {
            data_t sqrt3d = std::sqrt(static_cast<data_t>(3));

            Eigen::Matrix<data_t, volSize * volSize * volSize, 1> backProj;

            auto gamma = Gamma{Degree{30}};

            WHEN("A ray goes through the center of the volume")
            {
                // In this case the ray enters and exits the volume along the main direction
                geom.emplace_back(stc, ctr, std::move(volData), std::move(sinoData),
                                  RotationAngles3D{gamma});

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer<data_t> sino(sinoDescriptor);

                TestType op(volumeDescriptor, sinoDescriptor);

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
                        REQUIRE_EQ(sino[0], Approx(3 * sqrt3d - 1).epsilon(epsilon));

                        sino[0] = 1;
                        // clang-format off
                        backProj << 0, 0, 0, 0, 1 - 1 / sqrt3d, sqrt3d - 1, 0, 0, 0,
                                    0, 0, 0, 0, 2 / sqrt3d, 0, 0, 0, 0,
                                    0, 0, 0, sqrt3d - 1, 1 - 1 / sqrt3d, 0, 0, 0, 0;
                        // clang-format on

                        op.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(isApprox(
                            volume, DataContainer<data_t>(volumeDescriptor, backProj), epsilon));
                    }
                }
            }

            WHEN("A ray with an angle of 30 degrees enters through the right border")
            {
                // In this case the ray enters through a border orthogonal to a non-main
                // direction
                geom.emplace_back(stc, ctr, std::move(volData), std::move(sinoData),
                                  RotationAngles3D{Gamma{pi_t / 6}}, PrincipalPointOffset2D{0, 0},
                                  RotationOffset3D{1, 0, 0});

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer<data_t> sino(sinoDescriptor);

                TestType op(volumeDescriptor, sinoDescriptor);

                THEN("The ray intersects the correct voxels")
                {
                    volume = 1;
                    volume(2, 1, 1) = 0;
                    volume(2, 1, 0) = 0;
                    volume(2, 1, 2) = 0;
                    volume(1, 1, 2) = 0;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0], Approx(0).epsilon(epsilon));

                    AND_THEN("The correct weighting is applied")
                    {
                        volume(2, 1, 0) = 4;
                        volume(1, 1, 2) = 3;
                        volume(2, 1, 1) = 1;

                        op.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx(1 - 2 / sqrt3d + 3 * sqrt3d));

                        sino[0] = 1;

                        // clang-format off
                        backProj << 0, 0, 0, 0, 0, 1 - 1 / sqrt3d, 0, 0, 0,
                            0, 0, 0, 0, 0, 2 / sqrt3d, 0, 0, 0,
                            0, 0, 0, 0, sqrt3d - 1, 1 - 1 / sqrt3d, 0, 0, 0;
                        // clang-format on

                        op.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(isApprox(
                            volume, DataContainer<data_t>(volumeDescriptor, backProj), epsilon));
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
                DataContainer<data_t> sino(sinoDescriptor);

                TestType op(volumeDescriptor, sinoDescriptor);

                THEN("The ray intersects the correct voxels")
                {
                    volume = 1;
                    volume(0, 1, 0) = 0;
                    volume(1, 1, 0) = 0;
                    volume(0, 1, 1) = 0;
                    volume(0, 1, 2) = 0;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0], Approx(0).epsilon(epsilon));

                    AND_THEN("The correct weighting is applied")
                    {
                        volume(0, 1, 2) = 4;
                        volume(1, 1, 0) = 3;
                        volume(0, 1, 1) = 1;

                        op.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx(3 * sqrt3d + 1 - 2 / sqrt3d).epsilon(epsilon));

                        sino[0] = 1;

                        // clang-format off
                        backProj << 0, 0, 0, 1 - 1 / sqrt3d, sqrt3d - 1, 0, 0, 0, 0,
                                    0, 0, 0, 2 / sqrt3d, 0, 0, 0, 0, 0,
                                    0, 0, 0, 1 - 1 / sqrt3d, 0, 0, 0, 0, 0;
                        // clang-format on

                        op.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(isApprox(
                            volume, DataContainer<data_t>(volumeDescriptor, backProj), epsilon));
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
                DataContainer<data_t> sino(sinoDescriptor);

                TestType op(volumeDescriptor, sinoDescriptor);

                THEN("The ray intersects the correct voxels")
                {
                    volume = 1;
                    volume(0, 1, 0) = 0;

                    op.apply(volume, sino);
                    REQUIRE_EQ(sino[0], Approx(0).epsilon(epsilon));

                    AND_THEN("The correct weighting is applied")
                    {
                        volume(0, 1, 0) = 1;

                        op.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx(sqrt3d - 1).epsilon(epsilon));

                        sino[0] = 1;

                        // clang-format off
                        backProj << 0, 0, 0, sqrt3d - 1, 0, 0, 0, 0, 0,
                                    0, 0, 0,          0, 0, 0, 0, 0, 0,
                                    0, 0, 0,          0, 0, 0, 0, 0, 0;
                        // clang-format on

                        op.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(isApprox(
                            volume, DataContainer<data_t>(volumeDescriptor, backProj), epsilon));
                    }
                }
            }
        }
    }

    GIVEN("Given a 5x5x5 volume")
    {
        // Domain setup
        const index_t volSize = 5;

        IndexVector_t volumeDims(3);
        volumeDims << volSize, volSize, volSize;

        VolumeDescriptor volumeDescriptor(volumeDims);
        DataContainer<data_t> volume(volumeDescriptor);

        // range setup
        const index_t detectorSize = 1;
        const index_t numImgs = 1;

        IndexVector_t sinoDims(3);
        sinoDims << detectorSize, detectorSize, numImgs;

        // Setup geometry
        auto stc = SourceToCenterOfRotation{20 * volSize};
        auto ctr = CenterOfRotationToDetector{volSize};
        auto volData = VolumeData3D{Size3D{volumeDims}};
        auto sinoData = SinogramData3D{Size3D{sinoDims}};

        std::vector<Geometry> geom;

        GIVEN("Tracing rays along axis")
        {
            const index_t numCases = 9;
            using Array = std::array<real_t, numCases>;
            using StrArray = std::array<std::string, numCases>;

            Array alpha = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            Array beta = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, pi_t / 2, pi_t / 2, pi_t / 2};
            Array gamma = {0.0,      0.0,      0.0,      pi_t / 2, pi_t / 2,
                           pi_t / 2, pi_t / 2, pi_t / 2, pi_t / 2};

            Array offsetx = {volSize, 0.0, volSize, 0.0, 0.0, 0.0, volSize, 0.0, volSize};
            Array offsety = {0.0, volSize, volSize, volSize, 0.0, volSize, 0.0, 0.0, 0.0};
            Array offsetz = {0.0, 0.0, 0.0, 0.0, volSize, volSize, 0.0, volSize, volSize};

            StrArray neg = {"x", "y", "x and y", "y", "z", "y and z", "x", "z", "x and z"};
            StrArray ali = {"z", "z", "z", "x", "x", "x", "y", "y", "y"};

            for (unsigned i = 0; i < numCases; i++) {
                WHEN("Tracing along a fixed axis-aligned ray with negative coordinate of origin")
                {
                    INFO("Tracing along a ", ali[i], "-axis-aligned ray with negative ", neg[i],
                         "-coodinate of origin");
                    geom.emplace_back(
                        stc, ctr, std::move(volData), std::move(sinoData),
                        RotationAngles3D{Gamma{gamma[i]}, Beta{beta[i]}, Alpha{alpha[i]}},
                        PrincipalPointOffset2D{0, 0},
                        RotationOffset3D{-offsetx[i], -offsety[i], -offsetz[i]});

                    PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                    DataContainer<data_t> sino(sinoDescriptor);

                    TestType op(volumeDescriptor, sinoDescriptor);

                    THEN("Result of forward projection is zero")
                    {
                        op.apply(volume, sino);
                        DataContainer<data_t> zero(sinoDescriptor);
                        zero = 0;
                        REQUIRE_UNARY(isApprox(sino, zero, epsilon));

                        AND_THEN("Result of backprojection is zero")
                        {
                            op.applyAdjoint(sino, volume);
                            DataContainer<data_t> zero(volumeDescriptor);
                            zero = 0;
                            REQUIRE_UNARY(isApprox(volume, zero, epsilon));
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE_TEMPLATE("SiddonsMethodCUDA: 3D setup with a multiple rays", TestType,
                   SiddonsMethodCUDA<float>, SiddonsMethodCUDA<double>, SiddonsMethod<float>,
                   SiddonsMethod<double>)
{
    // Turn logger of
    Logger::setLevel(Logger::LogLevel::OFF);

    using data_t = decltype(return_data_t(std::declval<TestType>()));

    GIVEN("A 3D setting with multiple projection angles")
    {
        // Domain setup
        const index_t volSize = 3;

        IndexVector_t volumeDims(3);
        volumeDims << volSize, volSize, volSize;

        VolumeDescriptor volumeDescriptor(volumeDims);
        DataContainer<data_t> volume(volumeDescriptor);

        // range setup
        const index_t detectorSize = 1;
        const index_t numImgs = 6;

        IndexVector_t sinoDims(3);
        sinoDims << detectorSize, detectorSize, numImgs;

        // Setup geometry
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
            DataContainer<data_t> sino(sinoDescriptor);

            TestType op(volumeDescriptor, sinoDescriptor);

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
                    Eigen::Matrix<data_t, Eigen::Dynamic, 1> cmp(volSize * volSize * volSize);

                    // clang-format off
                    cmp << 0, 0, 0, 0,  6, 0, 0, 0, 0,
                           0, 6, 0, 6, 18, 6, 0, 6, 0,
                           0, 0, 0, 0,  6, 0, 0, 0, 0;
                    // clang-format on

                    op.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isApprox(volume, DataContainer<data_t>(volumeDescriptor, cmp)));
                }
            }
        }
    }
}
