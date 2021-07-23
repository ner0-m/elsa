#include "doctest/doctest.h"

#include "JosephsMethodCUDA.h"
#include "Geometry.h"
#include "VolumeDescriptor.h"
#include "Logger.h"
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

TEST_CASE_TEMPLATE("JosephsMethodCUDA: Calls to functions of super class", TestType,
                   JosephsMethodCUDA<float>, JosephsMethodCUDA<double>)
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
        volume = 0;

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

        TestType fast(volumeDescriptor, sinoDescriptor);
        TestType slow(volumeDescriptor, sinoDescriptor, false);

        WHEN("Projector is cloned")
        {
            auto fastClone = fast.clone();
            auto slowClone = slow.clone();
            auto sinoClone = DataContainer<data_t>(sinoDescriptor);
            sinoClone = 0;
            auto volumeClone = DataContainer<data_t>(volumeDescriptor);
            volumeClone = 0;

            THEN("Results do not change (may still be slightly different due to summation being "
                 "performed in a different order)")
            {
                fast.apply(volume, sino);
                fastClone->apply(volume, sinoClone);
                REQUIRE_UNARY(isCwiseApprox(sino, sinoClone));

                slowClone->apply(volume, sinoClone);
                REQUIRE_UNARY(isCwiseApprox(sino, sinoClone));

                fast.applyAdjoint(sino, volume);
                fastClone->applyAdjoint(sino, volumeClone);
                REQUIRE_UNARY(isCwiseApprox(volume, volumeClone));

                slow.applyAdjoint(sino, volume);
                slowClone->applyAdjoint(sino, volumeClone);
                REQUIRE_UNARY(isCwiseApprox(volume, volumeClone));
            }
        }
    }
}

TEST_CASE_TEMPLATE("JosephsMethodCUDA: 2D setup with a single ray", TestType,
                   JosephsMethodCUDA<float>, JosephsMethodCUDA<double>)
{
    // Turn logger of
    Logger::setLevel(Logger::LogLevel::OFF);

    using data_t = decltype(return_data_t(std::declval<TestType>()));

    GIVEN("A 5x5 volume and detector of size 1 with 1 angle")
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

            TestType slow(volumeDescriptor, sinoDescriptor, false);
            TestType fast(volumeDescriptor, sinoDescriptor);

            WHEN("Sinogram conatainer is not zero initialized and we project through an empty "
                 "volume")
            {
                volume = 0;
                sino = 1;

                THEN("Result is zero")
                {
                    fast.apply(volume, sino);
                    DataContainer<data_t> zero(sinoDescriptor);
                    zero = 0;
                    REQUIRE_UNARY(isCwiseApprox(sino, zero));

                    sino = 1;
                    slow.apply(volume, sino);
                    REQUIRE_UNARY(isCwiseApprox(sino, zero));
                }
            }

            WHEN("Volume container is not zero initialized and we backproject from an empty "
                 "sinogram")
            {
                sino = 0;
                volume = 1;

                THEN("Result is zero")
                {
                    fast.applyAdjoint(sino, volume);
                    DataContainer<data_t> zero(volumeDescriptor);
                    zero = 0;
                    REQUIRE_UNARY(isCwiseApprox(volume, zero));

                    volume = 1;
                    slow.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isCwiseApprox(volume, zero));
                }
            }
        }

        GIVEN("Rays not intersecting the bounding box are present")
        {
            WHEN("Tracing along a y-axis-aligned ray with a negative x-coordinate of origin")
            {
                geom.emplace_back(stc, ctr, Radian{0}, std::move(volData), std::move(sinoData),
                                  PrincipalPointOffset{}, RotationOffset2D{volSize, 0});

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer<data_t> sino(sinoDescriptor);

                TestType fast(volumeDescriptor, sinoDescriptor);
                TestType slow(volumeDescriptor, sinoDescriptor, false);

                THEN("Result of forward projection is zero")
                {
                    fast.apply(volume, sino);
                    DataContainer<data_t> zero(sinoDescriptor);
                    zero = 0;
                    REQUIRE_UNARY(isCwiseApprox(sino, zero));

                    slow.apply(volume, sino);
                    REQUIRE_UNARY(isCwiseApprox(sino, zero));

                    AND_THEN("Result of backprojection is zero")
                    {
                        fast.applyAdjoint(sino, volume);
                        DataContainer<data_t> zero(volumeDescriptor);
                        zero = 0;
                        REQUIRE_UNARY(isCwiseApprox(volume, zero));

                        slow.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(isCwiseApprox(volume, zero));
                    }
                }
            }

            WHEN("Tracing along a y-axis-aligned ray with a x-coordinate of origin beyond the "
                 "bounding "
                 "box")
            {
                geom.emplace_back(stc, ctr, Radian{0}, std::move(volData), std::move(sinoData),
                                  PrincipalPointOffset{}, RotationOffset2D{-volSize, 0});

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer<data_t> sino(sinoDescriptor);

                TestType fast(volumeDescriptor, sinoDescriptor);
                TestType slow(volumeDescriptor, sinoDescriptor, false);

                THEN("Result of forward projection is zero")
                {
                    DataContainer<data_t> zero(sinoDescriptor);
                    zero = 0;

                    fast.apply(volume, sino);
                    REQUIRE_UNARY(isCwiseApprox(sino, zero));

                    slow.apply(volume, sino);
                    REQUIRE_UNARY(isCwiseApprox(sino, zero));

                    AND_THEN("Result of backprojection is zero")
                    {
                        DataContainer<data_t> zero(volumeDescriptor);
                        zero = 0;

                        fast.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(isCwiseApprox(volume, zero));

                        slow.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(isCwiseApprox(volume, zero));
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

                TestType fast(volumeDescriptor, sinoDescriptor);
                TestType slow(volumeDescriptor, sinoDescriptor, false);

                THEN("Result of forward projection is zero")
                {
                    DataContainer<data_t> zero(sinoDescriptor);
                    zero = 0;

                    fast.apply(volume, sino);
                    REQUIRE_UNARY(isCwiseApprox(sino, zero));

                    slow.apply(volume, sino);
                    REQUIRE_UNARY(isCwiseApprox(sino, zero));

                    AND_THEN("Result of backprojection is zero")
                    {
                        DataContainer<data_t> zero(volumeDescriptor);
                        zero = 0;

                        fast.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(isCwiseApprox(volume, zero));

                        slow.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(isCwiseApprox(volume, zero));
                    }
                }
            }

            WHEN("Tracing along a x-axis-aligned ray with a y-coordinate of origin beyond the "
                 "bounding "
                 "box")
            {
                geom.emplace_back(stc, ctr, Radian{pi_t / 2}, std::move(volData),
                                  std::move(sinoData), PrincipalPointOffset{},
                                  RotationOffset2D{0, -volSize});

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer<data_t> sino(sinoDescriptor);

                TestType fast(volumeDescriptor, sinoDescriptor);
                TestType slow(volumeDescriptor, sinoDescriptor, false);

                THEN("Result of forward projection is zero")
                {
                    DataContainer<data_t> zero(sinoDescriptor);
                    zero = 0;

                    fast.apply(volume, sino);
                    REQUIRE_UNARY(isCwiseApprox(sino, zero));

                    slow.apply(volume, sino);
                    REQUIRE_UNARY(isCwiseApprox(sino, zero));

                    AND_THEN("Result of backprojection is zero")
                    {
                        DataContainer<data_t> zero(volumeDescriptor);
                        zero = 0;

                        fast.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(isCwiseApprox(volume, zero));

                        slow.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(isCwiseApprox(volume, zero));
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

        // Testing for axis-aligned rays
        GIVEN("Given rays along the axes")
        {

            backProjections[0] << 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                1, 0, 0;
            backProjections[1] << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0;

            for (std::size_t i = 0; i < numCases; i++) {
                WHEN("An axis-aligned ray with a fixed angle passes through the center of a pixel")
                {
                    INFO("An axis-aligned ray with an angle of ", angles[i],
                         " radians passes through the center of a pixel");
                    geom.emplace_back(stc, ctr, Degree{angles[i]}, std::move(volData),
                                      std::move(sinoData));

                    PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                    DataContainer<data_t> sino(sinoDescriptor);

                    TestType fast(volumeDescriptor, sinoDescriptor);
                    TestType slow(volumeDescriptor, sinoDescriptor, false);

                    THEN("The result of projecting through a pixel is exactly the pixel value")
                    {
                        for (index_t j = 0; j < volSize; j++) {
                            volume = 0;
                            if (i % 2 == 0)
                                volume(volSize / 2, j) = 1;
                            else
                                volume(j, volSize / 2) = 1;

                            fast.apply(volume, sino);
                            // Using doubles significantly increases interpolation accuracy
                            // For example: when using floats, points very near the border (less
                            // than ~1/500th of a pixel away from the border) are rounded to
                            // actually lie on the border. This then yields more accurate results
                            // when using floats in some of the axis-aligned test cases, despite the
                            // lower interpolation accuracy.
                            //
                            //  => different requirements for floats and doubles, looser
                            // requirements for doubles
                            REQUIRE_EQ(sino[0], Approx(1.0));

                            slow.apply(volume, sino);
                            REQUIRE_EQ(sino[0], Approx(1.0));
                        }

                        AND_THEN(
                            "The backprojection sets the values of all hit pixels to the detector "
                            "value")
                        {
                            fast.applyAdjoint(sino, volume);
                            REQUIRE_UNARY(isCwiseApprox(
                                volume,
                                DataContainer<data_t>(volumeDescriptor, backProjections[i % 2])));

                            slow.applyAdjoint(sino, volume);
                            REQUIRE_UNARY(isCwiseApprox(
                                volume,
                                DataContainer<data_t>(volumeDescriptor, backProjections[i % 2])));
                        }
                    }
                }
            }
        }

        // Move the source far far back, so rays become nearly parallel
        stc = SourceToCenterOfRotation{volSize * 2000};

        GIVEN("Off center rays along axes")
        {
            std::array<real_t, numCases> offsetx = {-0.25, 0.0, -0.25, 0.0};
            std::array<real_t, numCases> offsety = {0.0, -0.25, 0.0, -0.25};

            backProjections[0] << 0, 0.25, 0.75, 0, 0, 0, 0.25, 0.75, 0, 0, 0, 0.25, 0.75, 0, 0, 0,
                0.25, 0.75, 0, 0, 0, 0.25, 0.75, 0, 0;

            backProjections[1] << 0, 0, 0, 0, 0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75,
                0.75, 0.75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

            for (std::size_t i = 0; i < numCases; i++) {
                WHEN("An axis-aligned ray with a fixed angle, which does not pass through the "
                     "center of a pixel")
                {
                    INFO("An axis-aligned ray with an angle of ", angles[i],
                         " radians not passing through the center of a pixel");
                    geom.emplace_back(stc, ctr, Degree{angles[i]}, std::move(volData),
                                      std::move(sinoData), PrincipalPointOffset{0},
                                      RotationOffset2D{offsetx[i], offsety[i]});

                    PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                    DataContainer<data_t> sino(sinoDescriptor);

                    TestType fast(volumeDescriptor, sinoDescriptor);
                    TestType slow(volumeDescriptor, sinoDescriptor, false);

                    THEN("The result of projecting through a pixel is the interpolated value "
                         "between "
                         "the two pixels closest to the ray")
                    {
                        for (index_t j = 0; j < volSize; j++) {
                            volume = 0;
                            if (i % 2 == 0)
                                volume(volSize / 2, j) = 1;
                            else
                                volume(j, volSize / 2) = 1;

                            fast.apply(volume, sino);
                            REQUIRE_EQ(sino[0], Approx(0.75));

                            slow.apply(volume, sino);
                            REQUIRE_EQ(sino[0], Approx(0.75));
                        }

                        AND_THEN("The slow backprojection yields the exact adjoint, the fast "
                                 "backprojection "
                                 "also yields the exact adjoint for a very distant x-ray source")
                        {
                            sino[0] = 1;
                            slow.applyAdjoint(sino, volume);
                            REQUIRE_UNARY(isCwiseApprox(
                                volume,
                                DataContainer<data_t>(volumeDescriptor, backProjections[i % 2])));

                            fast.applyAdjoint(sino, volume);

                            // Using doubles significantly increases interpolation accuracy
                            //  For example: when using floats, points very near the border (less
                            //  than
                            // ~1/500th of a pixel away from the border) are rounded to actually lie
                            // on the border. This then yields more accurate results when using
                            // floats in some of the axis-aligned test cases, despite the lower
                            // interpolation accuracy.
                            //
                            //  => different requirements for floats and doubles, looser
                            //  requirements
                            // for doubles
                            if constexpr (std::is_same_v<data_t, float>)
                                REQUIRE_UNARY(isCwiseApprox(
                                    volume, DataContainer<data_t>(volumeDescriptor,
                                                                  backProjections[i % 2])));
                            else
                                REQUIRE_UNARY(isApprox(
                                    volume,
                                    DataContainer<data_t>(volumeDescriptor, backProjections[i % 2]),
                                    static_cast<real_t>(0.001)));
                        }
                    }
                }
            }
        }

        GIVEN("Rays going running allong right bolume boundary")
        {
            backProjections[0] << 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                0, 0, 1;

            WHEN("A y-axis-aligned ray runs along the right volume boundary")
            {
                geom.emplace_back(stc, ctr, Radian{0}, std::move(volData), std::move(sinoData),
                                  PrincipalPointOffset{0}, RotationOffset2D{volSize * 0.5, 0});

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer<data_t> sino(sinoDescriptor);

                TestType fast(volumeDescriptor, sinoDescriptor);
                TestType slow(volumeDescriptor, sinoDescriptor, false);

                THEN("The result of projecting through a pixel is exactly the pixel's value (we "
                     "mirror "
                     "values at the border for the purpose of interpolation)")
                {
                    for (index_t j = 0; j < volSize; j++) {
                        volume = 0;
                        volume(volSize - 1, j) = 1;

                        fast.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx(1));

                        slow.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx(1));
                    }

                    AND_THEN(
                        "The slow backprojection yields the exact adjoint, the fast backprojection "
                        "also yields the exact adjoint for a very distant x-ray source")
                    {
                        sino[0] = 1;
                        slow.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(isCwiseApprox(
                            volume, DataContainer<data_t>(volumeDescriptor, backProjections[0])));

                        fast.applyAdjoint(sino, volume);
                        // Using doubles significantly increases interpolation accuracy
                        //  For example: when using floats, points very near the border (less than
                        // ~1/500th of a pixel away from the border) are rounded to actually lie on
                        // the border. This then yields more accurate results when using floats in
                        // some of the axis-aligned test cases, despite the lower interpolation
                        // accuracy.
                        //
                        //  => different requirements for floats and doubles, looser requirements
                        // for doubles
                        if constexpr (std::is_same_v<data_t, float>)
                            REQUIRE_UNARY(isCwiseApprox(
                                volume, DataContainer<data_t>(volumeDescriptor,
                                                              (backProjections[0] / 2).eval())));
                        else
                            REQUIRE_UNARY(
                                isApprox(volume,
                                         DataContainer<data_t>(volumeDescriptor,
                                                               (backProjections[0] / 2).eval()),
                                         static_cast<real_t>(0.001)));
                    }
                }
            }
        }

        GIVEN("Rays going running along left bolume boundary")
        {
            backProjections[0] << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                0, 0, 0;

            WHEN("A y-axis-aligned ray runs along the left volume boundary")
            {
                geom.emplace_back(stc, ctr, Radian{0}, std::move(volData), std::move(sinoData),
                                  PrincipalPointOffset{0}, RotationOffset2D{-volSize * 0.5, 0});

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer<data_t> sino(sinoDescriptor);

                TestType fast(volumeDescriptor, sinoDescriptor);
                TestType slow(volumeDescriptor, sinoDescriptor, false);

                // TestType fast(volumeDescriptor, sinoDescriptor, geom);
                // TestType slow(volumeDescriptor, sinoDescriptor, geom, false);
                THEN("The result of projecting through a pixel is exactly the pixel's value (we "
                     "mirror "
                     "values at the border for the purpose of interpolation)")
                {
                    for (index_t j = 0; j < volSize; j++) {
                        volume = 0;
                        volume(0, j) = 1;

                        fast.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx(1));

                        slow.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx(1));
                    }

                    AND_THEN(
                        "The slow backprojection yields the exact adjoint, the fast backprojection "
                        "also yields the exact adjoint for a very distant x-ray source")
                    {
                        sino[0] = 1;
                        slow.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(isCwiseApprox(
                            volume, DataContainer<data_t>(volumeDescriptor, backProjections[0])));

                        fast.applyAdjoint(sino, volume);
                        /** Using doubles significantly increases interpolation accuracy
                         *  For example: when using floats, points very near the border (less than
                         * ~1/500th of a pixel away from the border) are rounded to actually lie on
                         * the border. This then yields more accurate results when using floats in
                         * some of the axis-aligned test cases, despite the lower interpolation
                         * accuracy.
                         *
                         *  => different requirements for floats and doubles, looser requirements
                         * for doubles
                         */
                        if constexpr (std::is_same_v<data_t, float>)
                            REQUIRE_UNARY(isCwiseApprox(
                                volume, DataContainer<data_t>(volumeDescriptor,
                                                              (backProjections[0] / 2).eval())));
                        else
                            REQUIRE_UNARY(
                                isApprox(volume,
                                         DataContainer<data_t>(volumeDescriptor,
                                                               (backProjections[0] / 2).eval()),
                                         static_cast<real_t>(0.001)));
                    }
                }
            }
        }
    }

    GIVEN("A 4x4 volume and detector of size 1 with 1 angle at -30 degree")
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
        const auto stc = SourceToCenterOfRotation{20 * volSize};
        const auto ctr = CenterOfRotationToDetector{volSize};
        auto volData = VolumeData2D{Size2D{volumeDims}};
        auto sinoData = SinogramData2D{Size2D{sinoDims}};

        std::vector<Geometry> geom;

        const real_t sqrt3r = std::sqrt(static_cast<real_t>(3));
        const data_t sqrt3d = std::sqrt(static_cast<data_t>(3));
        const data_t halfd = static_cast<data_t>(0.5);
        const data_t thirdd = static_cast<data_t>(1.0 / 3);

        GIVEN("An angle of -30 degrees")
        {
            const auto angle = Degree{-30};

            WHEN("Projections goes through center of volume")
            {
                geom.emplace_back(stc, ctr, angle, std::move(volData), std::move(sinoData));

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer<data_t> sino(sinoDescriptor);
                sino = 0;

                TestType fast(volumeDescriptor, sinoDescriptor);
                TestType slow(volumeDescriptor, sinoDescriptor, false);

                real_t weight = 2 / sqrt3r;

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

                    fast.apply(volume, sino);
                    DataContainer<data_t> zero(sinoDescriptor);
                    zero = 0;
                    REQUIRE_UNARY(isCwiseApprox(sino, zero));

                    slow.apply(volume, sino);
                    REQUIRE_UNARY(isCwiseApprox(sino, zero));

                    AND_THEN("The correct weighting is applied")
                    {
                        volume(3, 0) = 1;
                        volume(1, 1) = 1;
                        volume(1, 2) = 1;
                        volume(1, 3) = 1;

                        fast.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx(2 * weight));

                        slow.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx(2 * weight));

                        sino[0] = 1;

                        Eigen::Matrix<data_t, Eigen::Dynamic, 1> slowExpected(volSize * volSize);
                        slowExpected << 0, 0, (3 - sqrt3d) / 2, (sqrt3d - 1) / 2, 0,
                            (sqrt3d - 1) / (2 * sqrt3d), (sqrt3d + 1) / (2 * sqrt3d), 0, 0,
                            (sqrt3d + 1) / (2 * sqrt3d), (sqrt3d - 1) / (2 * sqrt3d), 0,
                            (sqrt3d - 1) / 2, (3 - sqrt3d) / 2, 0, 0;

                        slowExpected *= weight;
                        slow.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(isCwiseApprox(
                            volume, DataContainer<data_t>(volumeDescriptor, slowExpected)));

                        fast.applyAdjoint(sino, volume);
                        for (real_t i = 0.5; i < volSize; i += 1) {
                            for (real_t j = 0.5; j < volSize; j += 1) {
                                const real_t angle =
                                    std::abs(std::atan((sqrt3r * volSize * 10
                                                        - static_cast<real_t>(volSize / 2.0) + j)
                                                       / (volSize * 10
                                                          + static_cast<real_t>(volSize / 2.0) - i))
                                             - pi_t / 3);
                                const real_t len = volSize * 21 * std::tan(angle);
                                if (len < 1) {
                                    REQUIRE_EQ(volume((index_t) i, (index_t) j),
                                               Approx(1 - len).epsilon(0.005));
                                } else {
                                    REQUIRE_EQ(volume((index_t) i, (index_t) j), Approx(0));
                                }
                            }
                        }
                    }
                }
            }

            WHEN("Ray enters through the right border")
            {
                // In this case the ray exits through a border along the main ray direction, but
                // enters through a border not along the main direction First pixel should be
                // weighted differently
                geom.emplace_back(stc, ctr, angle, std::move(volData), std::move(sinoData),
                                  PrincipalPointOffset{0}, RotationOffset2D{sqrt3r, 0});

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer<data_t> sino(sinoDescriptor);
                sino = 0;

                TestType fast(volumeDescriptor, sinoDescriptor);
                TestType slow(volumeDescriptor, sinoDescriptor, false);

                // TestType fast(volumeDescriptor, sinoDescriptor, geom);
                // TestType slow(volumeDescriptor, sinoDescriptor, geom, false);

                THEN("Ray intersects the correct pixels")
                {
                    volume = 1;
                    volume(3, 1) = 0;
                    volume(3, 2) = 0;
                    volume(3, 3) = 0;
                    volume(2, 3) = 0;
                    volume(2, 2) = 0;

                    fast.apply(volume, sino);
                    DataContainer<data_t> zero(sinoDescriptor);
                    zero = 0;
                    REQUIRE_UNARY(isCwiseApprox(sino, zero));

                    slow.apply(volume, sino);
                    REQUIRE_UNARY(isCwiseApprox(sino, zero));

                    AND_THEN("The correct weighting is applied")
                    {
                        volume(3, 1) = 1;
                        volume(2, 2) = 1;
                        volume(2, 3) = 1;

                        fast.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx((4 - 2 * sqrt3d) * (sqrt3d - 1)
                                                   + (2 / sqrt3d) * (3 - 8 * sqrt3d / 6))
                                                .epsilon(0.005));

                        slow.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx((4 - 2 * sqrt3d) * (sqrt3d - 1)
                                                   + (2 / sqrt3d) * (3 - 8 * sqrt3d / 6))
                                                .epsilon(0.005));

                        sino[0] = 1;

                        Eigen::Matrix<data_t, Eigen::Dynamic, 1> slowExpected(volSize * volSize);
                        slowExpected << 0, 0, 0, 0, 0, 0, 0, (4 - 2 * sqrt3d) * (sqrt3d - 1), 0, 0,
                            (2 / sqrt3d) * (1 + halfd - 5 * sqrt3d / 6),
                            (4 - 2 * sqrt3d) * (2 - sqrt3d)
                                + (2 / sqrt3d) * (5 * sqrt3d / 6 - halfd),
                            0, 0, (2 / sqrt3d) * (1 + halfd - sqrt3d / 2),
                            (2 / sqrt3d) * (sqrt3d / 2 - halfd);

                        slow.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(isCwiseApprox(
                            volume, DataContainer<data_t>(volumeDescriptor, slowExpected)));

                        fast.applyAdjoint(sino, volume);
                        for (real_t i = 0.5; i < volSize; i += 1) {
                            for (real_t j = 0.5; j < volSize; j += 1) {
                                const real_t angle =
                                    std::abs(std::atan((40 * sqrt3r - 2 + j) / (42 + sqrt3r - i))
                                             - pi_t / 3);
                                const real_t len = 84 * std::tan(angle);
                                if (len < 1) {
                                    REQUIRE_EQ(volume((index_t) i, (index_t) j),
                                               Approx(1 - len).epsilon(0.01));
                                } else {
                                    REQUIRE_EQ(volume((index_t) i, (index_t) j), Approx(0));
                                }
                            }
                        }
                    }
                }
            }

            WHEN("Ray exits through the left border")
            {
                // In this case the ray enters through a border along the main ray direction, but
                // exits through a border not along the main direction Last pixel should be weighted
                // differently
                geom.emplace_back(stc, ctr, angle, std::move(volData), std::move(sinoData),
                                  PrincipalPointOffset{0}, RotationOffset2D{-sqrt3r, 0});

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer<data_t> sino(sinoDescriptor);
                sino = 0;

                TestType fast(volumeDescriptor, sinoDescriptor);
                TestType slow(volumeDescriptor, sinoDescriptor, false);

                THEN("Ray intersects the correct pixels")
                {
                    volume = 1;
                    volume(0, 0) = 0;
                    volume(1, 0) = 0;
                    volume(0, 1) = 0;
                    volume(1, 1) = 0;
                    volume(0, 2) = 0;

                    fast.apply(volume, sino);
                    DataContainer<data_t> zero(sinoDescriptor);
                    zero = 0;
                    REQUIRE_UNARY(isCwiseApprox(sino, zero));

                    slow.apply(volume, sino);
                    REQUIRE_UNARY(isCwiseApprox(sino, zero));

                    AND_THEN("The correct weighting is applied")
                    {
                        volume(1, 0) = 1;
                        volume(0, 1) = 1;

                        fast.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx((sqrt3d - 1) + (5.0 / 3.0 - 1 / sqrt3d)
                                                   + (4 - 2 * sqrt3d) * (2 - sqrt3d))
                                                .epsilon(0.005));

                        slow.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx((sqrt3d - 1) + (5.0 / 3.0 - 1 / sqrt3d)
                                                   + (4 - 2 * sqrt3d) * (2 - sqrt3d))
                                                .epsilon(0.005));

                        sino[0] = 1;

                        Eigen::Matrix<data_t, Eigen::Dynamic, 1> slowExpected(volSize * volSize);
                        slowExpected << 1 - 1 / sqrt3d, sqrt3d - 1, 0, 0,
                            (5 * thirdd - 1 / sqrt3d) + (4 - 2 * sqrt3d) * (2 - sqrt3d),
                            sqrt3d - 5 * thirdd, 0, 0, (sqrt3d - 1) * (4 - 2 * sqrt3d), 0, 0, 0, 0,
                            0, 0, 0;

                        slow.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(isCwiseApprox(
                            volume, DataContainer<data_t>(volumeDescriptor, slowExpected)));

                        fast.applyAdjoint(sino, volume);

                        for (real_t i = 0.5; i < volSize; i += 1) {
                            for (real_t j = 0.5; j < volSize; j += 1) {
                                const real_t angle =
                                    std::abs(std::atan((40 * sqrt3r - 2 + j) / (42 - sqrt3r - i))
                                             - pi_t / 3);
                                const real_t len = 84 * std::tan(angle);

                                if (len < 1) {
                                    REQUIRE_EQ(volume((index_t) i, (index_t) j),
                                               Approx(1 - len).epsilon(0.002));
                                } else {
                                    REQUIRE_EQ(volume((index_t) i, (index_t) j), Approx(0));
                                }
                            }
                        }
                    }
                }
            }

            WHEN("Ray only intersects a single pixel")
            {
                // This is a special case that is handled separately in both forward and
                // backprojection
                geom.emplace_back(stc, ctr, angle, std::move(volData), std::move(sinoData),
                                  PrincipalPointOffset{0}, RotationOffset2D{-2 - sqrt3r / 2, 0});

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer<data_t> sino(sinoDescriptor);
                sino = 0;

                TestType fast(volumeDescriptor, sinoDescriptor);
                TestType slow(volumeDescriptor, sinoDescriptor, false);

                THEN("Ray intersects the correct pixels")
                {
                    volume = 1;
                    volume(0, 0) = 0;

                    fast.apply(volume, sino);
                    DataContainer<data_t> zero(sinoDescriptor);
                    zero = 0;
                    REQUIRE_UNARY(isCwiseApprox(sino, zero));

                    slow.apply(volume, sino);
                    REQUIRE_UNARY(isCwiseApprox(sino, zero));

                    AND_THEN("The correct weighting is applied")
                    {
                        volume(0, 0) = 1;

                        fast.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx(1 / sqrt3d).epsilon(0.005));

                        slow.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx(1 / sqrt3d).epsilon(0.005));

                        sino[0] = 1;

                        Eigen::Matrix<data_t, Eigen::Dynamic, 1> slowExpected(volSize * volSize);
                        slowExpected << 1 / sqrt3d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

                        slow.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(isCwiseApprox(
                            volume, DataContainer<data_t>(volumeDescriptor, slowExpected)));

                        fast.applyAdjoint(sino, volume);

                        for (real_t i = 0.5; i < volSize; i += 1) {
                            for (real_t j = 0.5; j < volSize; j += 1) {
                                const real_t angle = std::abs(
                                    std::atan((40 * sqrt3r - 2 + j) / (40 - sqrt3r / 2 - i))
                                    - pi_t / 3);
                                const real_t len = 84 * std::tan(angle);

                                if (len < 1) {
                                    REQUIRE_EQ(volume((index_t) i, (index_t) j),
                                               Approx(1 - len).epsilon(0.002));
                                } else {
                                    REQUIRE_EQ(volume((index_t) i, (index_t) j), Approx(0));
                                }
                            }
                        }
                    }
                }
            }
        }

        GIVEN("An angle of -120 degrees")
        {
            const auto angle = Degree{-120};

            WHEN("Ray goes through center of volume")
            {
                // In this case the ray enters and exits the volume through the borders along the
                // main direction Weighting for all interpolated values should be the same
                geom.emplace_back(stc, ctr, angle, std::move(volData), std::move(sinoData));
                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer<data_t> sino(sinoDescriptor);
                sino = 0;

                TestType fast(volumeDescriptor, sinoDescriptor);
                TestType slow(volumeDescriptor, sinoDescriptor, false);

                real_t weight = 2 / sqrt3r;
                THEN("Ray intersects the correct pixels")
                {
                    sino[0] = 1;
                    slow.applyAdjoint(sino, volume);

                    volume = 1;
                    volume(0, 0) = 0;
                    volume(0, 1) = 0;
                    volume(1, 1) = 0;
                    volume(1, 2) = 0;

                    volume(2, 1) = 0;
                    volume(2, 2) = 0;
                    volume(3, 2) = 0;
                    volume(3, 3) = 0;

                    fast.apply(volume, sino);
                    DataContainer<data_t> zero(sinoDescriptor);
                    zero = 0;
                    REQUIRE_UNARY(isCwiseApprox(sino, zero));

                    slow.apply(volume, sino);
                    REQUIRE_UNARY(isCwiseApprox(sino, zero));

                    AND_THEN("The correct weighting is applied")
                    {
                        volume(3, 3) = 1;
                        volume(0, 1) = 1;
                        volume(1, 1) = 1;
                        volume(2, 1) = 1;

                        fast.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx(2 * weight));

                        slow.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx(2 * weight));

                        sino[0] = 1;

                        Eigen::Matrix<data_t, Eigen::Dynamic, 1> slowExpected(volSize * volSize);

                        slowExpected << (sqrt3d - 1) / 2, 0, 0, 0, (3 - sqrt3d) / 2,
                            (sqrt3d + 1) / (2 * sqrt3d), (sqrt3d - 1) / (2 * sqrt3d), 0, 0,
                            (sqrt3d - 1) / (2 * sqrt3d), (sqrt3d + 1) / (2 * sqrt3d),
                            (3 - sqrt3d) / 2, 0, 0, 0, (sqrt3d - 1) / 2;

                        slowExpected *= weight;
                        slow.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(isCwiseApprox(
                            volume, DataContainer<data_t>(volumeDescriptor, slowExpected)));

                        fast.applyAdjoint(sino, volume);
                        for (real_t i = 0.5; i < volSize; i += 1) {
                            for (real_t j = 0.5; j < volSize; j += 1) {
                                const real_t angle = std::abs(
                                    std::atan((sqrt3r * 40 + 2 - i) / (42 - j)) - pi_t / 3);
                                const real_t len = volSize * 21 * std::tan(angle);
                                if (len < 1) {
                                    REQUIRE_EQ(volume((index_t) i, (index_t) j),
                                               Approx(1 - len).epsilon(0.005));
                                } else {
                                    REQUIRE_EQ(volume((index_t) i, (index_t) j), Approx(0));
                                }
                            }
                        }
                    }
                }
            }

            WHEN("Ray enters through the top border")
            {
                // In this case the ray exits through a border along the main ray direction, but
                // enters through a border not along the main direction First pixel should be
                // weighted differently
                geom.emplace_back(stc, ctr, angle, std::move(volData), std::move(sinoData),
                                  PrincipalPointOffset{0}, RotationOffset2D{0, std::sqrt(3.f)});
                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer<data_t> sino(sinoDescriptor);
                sino = 0;

                TestType fast(volumeDescriptor, sinoDescriptor);
                TestType slow(volumeDescriptor, sinoDescriptor, false);

                THEN("Ray intersects the correct pixels")
                {
                    volume = 1;
                    volume(0, 2) = 0;
                    volume(0, 3) = 0;
                    volume(1, 2) = 0;
                    volume(1, 3) = 0;
                    volume(2, 3) = 0;

                    fast.apply(volume, sino);
                    DataContainer<data_t> zero(sinoDescriptor);
                    zero = 0;
                    REQUIRE_UNARY(isCwiseApprox(sino, zero));

                    slow.apply(volume, sino);
                    REQUIRE_UNARY(isCwiseApprox(sino, zero));

                    AND_THEN("The correct weighting is applied")
                    {
                        volume(2, 3) = 1;
                        volume(1, 2) = 1;
                        volume(1, 3) = 1;

                        fast.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx((4 - 2 * sqrt3d) + (2 / sqrt3d)));

                        slow.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx((4 - 2 * sqrt3d) + (2 / sqrt3d)));

                        sino[0] = 1;

                        Eigen::Matrix<data_t, Eigen::Dynamic, 1> slowExpected(volSize * volSize);

                        slowExpected << 0, 0, 0, 0, 0, 0, 0, 0,
                            (2 / sqrt3d) * (1 + halfd - sqrt3d / 2),
                            (2 / sqrt3d) * (1 + halfd - 5 * sqrt3d / 6), 0, 0,
                            (2 / sqrt3d) * (sqrt3d / 2 - halfd),
                            (4 - 2 * sqrt3d) * (2 - sqrt3d)
                                + (2 / sqrt3d) * (5 * sqrt3d / 6 - halfd),
                            (4 - 2 * sqrt3d) * (sqrt3d - 1), 0;

                        slow.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(isCwiseApprox(
                            volume, DataContainer<data_t>(volumeDescriptor, slowExpected)));

                        fast.applyAdjoint(sino, volume);
                        for (real_t i = 0.5; i < volSize; i += 1) {
                            for (real_t j = 0.5; j < volSize; j += 1) {
                                const real_t angle =
                                    std::abs(std::atan((sqrt3r * 40 + 2 - i) / (42 + sqrt3r - j))
                                             - pi_t / 3);
                                const real_t len = volSize * 21 * std::tan(angle);
                                if (len < 1) {
                                    REQUIRE_EQ(volume((index_t) i, (index_t) j),
                                               Approx(1 - len).epsilon(0.01));
                                } else {
                                    REQUIRE_EQ(volume((index_t) i, (index_t) j), Approx(0));
                                }
                            }
                        }
                    }
                }
            }

            WHEN("Ray exits through the bottom border")
            {
                // In this case the ray enters through a border along the main ray direction, but
                // exits through a border not along the main direction Last pixel should be weighted
                // differently
                geom.emplace_back(stc, ctr, angle, std::move(volData), std::move(sinoData),
                                  PrincipalPointOffset{0}, RotationOffset2D{0, -std::sqrt(3.f)});

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer<data_t> sino(sinoDescriptor);
                sino = 0;

                TestType fast(volumeDescriptor, sinoDescriptor);
                TestType slow(volumeDescriptor, sinoDescriptor, false);

                THEN("Ray intersects the correct pixels")
                {
                    volume = 1;
                    volume(1, 0) = 0;
                    volume(2, 0) = 0;
                    volume(3, 0) = 0;
                    volume(2, 1) = 0;
                    volume(3, 1) = 0;

                    fast.apply(volume, sino);
                    DataContainer<data_t> zero(sinoDescriptor);
                    zero = 0;
                    REQUIRE_UNARY(isCwiseApprox(sino, zero));

                    slow.apply(volume, sino);
                    REQUIRE_UNARY(isCwiseApprox(sino, zero));

                    AND_THEN("The correct weighting is applied")
                    {
                        volume(2, 0) = 1;
                        volume(3, 1) = 1;

                        fast.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx((sqrt3d - 1) + (5.0 / 3.0 - 1 / sqrt3d)
                                                   + (4 - 2 * sqrt3d) * (2 - sqrt3d))
                                                .epsilon(0.005));

                        slow.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx((sqrt3d - 1) + (5.0 / 3.0 - 1 / sqrt3d)
                                                   + (4 - 2 * sqrt3d) * (2 - sqrt3d))
                                                .epsilon(0.005));

                        sino[0] = 1;

                        Eigen::Matrix<data_t, Eigen::Dynamic, 1> slowExpected(volSize * volSize);

                        slowExpected << 0, (sqrt3d - 1) * (4 - 2 * sqrt3d),
                            (5 * thirdd - 1 / sqrt3d) + (4 - 2 * sqrt3d) * (2 - sqrt3d),
                            1 - 1 / sqrt3d, 0, 0, sqrt3d - 5 * thirdd, sqrt3d - 1, 0, 0, 0, 0, 0, 0,
                            0, 0;

                        slow.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(isCwiseApprox(
                            volume, DataContainer<data_t>(volumeDescriptor, slowExpected)));

                        fast.applyAdjoint(sino, volume);

                        for (real_t i = 0.5; i < volSize; i += 1) {
                            for (real_t j = 0.5; j < volSize; j += 1) {
                                const real_t angle =
                                    std::abs(std::atan((sqrt3r * 40 + 2 - i) / (42 - sqrt3r - j))
                                             - pi_t / 3);
                                const real_t len = 84 * std::tan(angle);

                                if (len < 1) {
                                    REQUIRE_EQ(volume((index_t) i, (index_t) j),
                                               Approx(1 - len).epsilon(0.002));
                                } else {
                                    REQUIRE_EQ(volume((index_t) i, (index_t) j), Approx(0));
                                }
                            }
                        }
                    }
                }
            }

            WHEN("Ray only intersects a single pixel")
            {
                // This is a special case that is handled separately in both forward and
                // backprojection
                geom.emplace_back(stc, ctr, angle, std::move(volData), std::move(sinoData),
                                  PrincipalPointOffset{0},
                                  RotationOffset2D{0, -2 - std::sqrt(3.f) / 2});

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer<data_t> sino(sinoDescriptor);
                sino = 0;

                TestType fast(volumeDescriptor, sinoDescriptor);
                TestType slow(volumeDescriptor, sinoDescriptor, false);

                THEN("Ray intersects the correct pixels")
                {
                    volume = 1;
                    volume(3, 0) = 0;

                    fast.apply(volume, sino);
                    DataContainer<data_t> zero(sinoDescriptor);
                    zero = 0;
                    REQUIRE_UNARY(isCwiseApprox(sino, zero));

                    slow.apply(volume, sino);
                    REQUIRE_UNARY(isCwiseApprox(sino, zero));

                    AND_THEN("The correct weighting is applied")
                    {
                        volume(3, 0) = 1;

                        fast.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx(1 / sqrt3d).epsilon(0.005));

                        slow.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx(1 / sqrt3d).epsilon(0.005));

                        sino[0] = 1;

                        Eigen::Matrix<data_t, Eigen::Dynamic, 1> slowExpected(volSize * volSize);
                        slowExpected << 0, 0, 0, 1 / sqrt3d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

                        slow.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(isCwiseApprox(
                            volume, DataContainer<data_t>(volumeDescriptor, slowExpected)));

                        fast.applyAdjoint(sino, volume);

                        for (real_t i = 0.5; i < volSize; i += 1) {
                            for (real_t j = 0.5; j < volSize; j += 1) {
                                const real_t angle = std::abs(
                                    std::atan((sqrt3r * 40 + 2 - i) / (40 - sqrt3r / 2 - j))
                                    - pi_t / 3);
                                const real_t len = 84 * std::tan(angle);

                                if (len < 1) {
                                    REQUIRE_EQ(volume((index_t) i, (index_t) j),
                                               Approx(1 - len).epsilon(0.002));
                                } else {
                                    REQUIRE_EQ(volume((index_t) i, (index_t) j), Approx(0));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE_TEMPLATE("JosephsMethodCUDA: 2D setup with a multiple rays", TestType,
                   JosephsMethodCUDA<float>, JosephsMethodCUDA<double>)
{
    // Turn logger of
    Logger::setLevel(Logger::LogLevel::OFF);

    using data_t = decltype(return_data_t(std::declval<TestType>()));

    GIVEN("a 5x5 Volume and detector of size 1, with 4 angles")
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
            sino = 0;

            TestType fast(volumeDescriptor, sinoDescriptor);
            TestType slow(volumeDescriptor, sinoDescriptor, false);

            THEN("Values are accumulated correctly along each ray's path")
            {
                volume = 0;

                // set only values along the rays' path to one to make sure interpolation is done
                // correctly
                for (index_t i = 0; i < volSize; i++) {
                    volume(i, volSize / 2) = 1;
                    volume(volSize / 2, i) = 1;
                }

                slow.apply(volume, sino);
                for (index_t i = 0; i < numImgs; i++)
                    REQUIRE_EQ(sino[i], Approx(5.0));

                fast.apply(volume, sino);
                for (index_t i = 0; i < numImgs; i++)
                    REQUIRE_EQ(sino[i], Approx(5.0));

                AND_THEN("Both fast and slow backprojections yield the exact adjoint")
                {
                    Eigen::Matrix<data_t, Eigen::Dynamic, 1> cmp(volSize * volSize);

                    cmp << 0, 0, 10, 0, 0, 0, 0, 10, 0, 0, 10, 10, 20, 10, 10, 0, 0, 10, 0, 0, 0, 0,
                        10, 0, 0;

                    slow.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(
                        isCwiseApprox(volume, DataContainer<data_t>(volumeDescriptor, cmp)));

                    fast.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(
                        isCwiseApprox(volume, DataContainer<data_t>(volumeDescriptor, cmp)));
                }
            }
        }
    }
}
TEST_CASE_TEMPLATE("JosephsMethodCUDA: 3D setup with a single ray", TestType,
                   JosephsMethodCUDA<float>, JosephsMethodCUDA<double>)
{
    // Turn logger of
    Logger::setLevel(Logger::LogLevel::OFF);

    using data_t = decltype(return_data_t(std::declval<TestType>()));

    GIVEN("A 3x3x3 Volume")
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
        const auto stc = SourceToCenterOfRotation{20 * volSize};
        const auto ctr = CenterOfRotationToDetector{volSize};
        auto volData = VolumeData3D{Size3D{volumeDims}};
        auto sinoData = SinogramData3D{Size3D{sinoDims}};

        std::vector<Geometry> geom;

        GIVEN("Single ray along z-axis")
        {

            geom.emplace_back(stc, ctr, std::move(volData), std::move(sinoData),
                              RotationAngles3D{Gamma{0}});

            PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
            DataContainer<data_t> sino(sinoDescriptor);
            sino = 0;

            TestType fast(volumeDescriptor, sinoDescriptor);
            TestType slow(volumeDescriptor, sinoDescriptor, false);

            WHEN("Sinogram conatainer is not zero initialized and we project through an empty "
                 "volume")
            {
                volume = 0;
                sino = 1;

                THEN("Result is zero")
                {
                    fast.apply(volume, sino);
                    DataContainer<data_t> zero(sinoDescriptor);
                    zero = 0;
                    REQUIRE_UNARY(isCwiseApprox(sino, zero));

                    sino = 1;
                    slow.apply(volume, sino);
                    REQUIRE_UNARY(isCwiseApprox(sino, zero));
                }
            }

            WHEN("Volume container is not zero initialized and we backproject from an empty "
                 "sinogram")
            {
                sino = 0;
                volume = 1;

                THEN("Result is zero")
                {
                    fast.applyAdjoint(sino, volume);
                    DataContainer<data_t> zero(volumeDescriptor);
                    zero = 0;
                    REQUIRE_UNARY(isCwiseApprox(volume, zero));

                    volume = 1;
                    slow.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(isCwiseApprox(volume, zero));
                }
            }
        }

        const index_t numCases = 6;
        std::array<real_t, numCases> beta = {0.0, 0.0, 0.0, 0.0, pi_t / 2, 3 * pi_t / 2};
        std::array<real_t, numCases> gamma = {0.0,          pi_t,     pi_t / 2,
                                              3 * pi_t / 2, pi_t / 2, 3 * pi_t / 2};
        std::array<std::string, numCases> al = {"z", "-z", "x", "-x", "y", "-y"};

        Eigen::Matrix<data_t, Eigen::Dynamic, 1> backProjections[numCases];
        for (auto& backPr : backProjections)
            backPr.resize(volSize * volSize * volSize);

        // clang-format off
        backProjections[0] << 0, 0, 0, 0, 1, 0, 0, 0, 0,
                              0, 0, 0, 0, 1, 0, 0, 0, 0,
                              0, 0, 0, 0, 1, 0, 0, 0, 0;

        backProjections[1] << 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 1, 1, 1, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0;

        backProjections[2] << 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 1, 0, 0, 1, 0, 0, 1, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0;
        // clang-format on

        for (std::size_t i = 0; i < numCases; i++) {
            WHEN("An axis-aligned ray passes through the center of a pixel")
            {
                INFO("A ", al[i], "-axis-aligned ray passes through the center of a pixel");
                geom.emplace_back(stc, ctr, std::move(volData), std::move(sinoData),
                                  RotationAngles3D{Gamma{gamma[i]}, Beta{beta[i]}});

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer<data_t> sino(sinoDescriptor);
                sino = 0;

                TestType fast(volumeDescriptor, sinoDescriptor);
                TestType slow(volumeDescriptor, sinoDescriptor, false);

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

                        fast.apply(volume, sino);
                        // Using doubles significantly increases interpolation accuracy
                        // For example: when using floats, points very near the border (less than
                        // ~1/500th of a pixel away from the border) are rounded to actually lie on
                        // the border. This then yields more accurate results when using floats in
                        // some of the axis-aligned test cases, despite the lower interpolation
                        // accuracy.
                        //  => different requirements for floats and doubles, looser requirements
                        // for doubles
                        REQUIRE_EQ(sino[0], Approx(1.0));

                        slow.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx(1.0));
                    }

                    AND_THEN("The backprojection sets the values of all hit voxels to the detector "
                             "value")
                    {
                        fast.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(
                            isCwiseApprox(volume, DataContainer<data_t>(volumeDescriptor,
                                                                        backProjections[i / 2])));

                        slow.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(
                            isCwiseApprox(volume, DataContainer<data_t>(volumeDescriptor,
                                                                        backProjections[i / 2])));
                    }
                }
            }
        }

        std::array<real_t, numCases> offsetx = {-0.25, -0.25, 0.0, 0.0, 0.0, 0.0};
        std::array<real_t, numCases> offsety = {0.0, 0.0, -0.25, -0.25, 0.0, 0.0};
        std::array<real_t, numCases> offsetz = {0.0, 0.0, 0.0, 0.0, -0.25, -0.25};

        // clang-format off
        backProjections[0] << 0, 0, 0, 0.25, 0.75, 0, 0, 0, 0,
                              0, 0, 0, 0.25, 0.75, 0, 0, 0, 0,
                              0, 0, 0, 0.25, 0.75, 0, 0, 0, 0;

        backProjections[1] <<  0   , 0   , 0   , 0   , 0   , 0   , 0, 0, 0,
                               0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0, 0, 0,
                               0   , 0   , 0   , 0   , 0   , 0   , 0, 0, 0;

        backProjections[2] << 0, 0.25, 0, 0, 0.25, 0, 0, 0.25, 0,
                              0, 0.75, 0, 0, 0.75, 0, 0, 0.75, 0,
                              0, 0   , 0, 0, 0   , 0, 0, 0   , 0;
        // clang-format on

        for (std::size_t i = 0; i < numCases; i++) {
            WHEN("An axis-aligned ray not passing through the center of a pixel")
            {
                INFO("A ", al[i], "-axis-aligned ray not passing through the center of a pixel");
                // x-ray source must be very far from the volume center to make testing of the fast
                // backprojection simpler
                geom.emplace_back(SourceToCenterOfRotation{volSize * 2000}, ctr, std::move(volData),
                                  std::move(sinoData),
                                  RotationAngles3D{Gamma{gamma[i]}, Beta{beta[i]}},
                                  PrincipalPointOffset2D{0, 0},
                                  RotationOffset3D{offsetx[i], offsety[i], offsetz[i]});

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer<data_t> sino(sinoDescriptor);
                sino = 0;

                TestType fast(volumeDescriptor, sinoDescriptor);
                TestType slow(volumeDescriptor, sinoDescriptor, false);

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

                        fast.apply(volume, sino);

                        // Using doubles significantly increases interpolation accuracy
                        //  For example: when using floats, points very near the border (less than
                        // ~1/500th of a pixel away from the border) are rounded to actually lie on
                        // the border. This then yields more accurate results when using floats in
                        // some of the axis-aligned test cases, despite the lower interpolation
                        // accuracy.
                        //
                        //  => different requirements for floats and doubles, looser requirements
                        // for doubles

                        REQUIRE_EQ(sino[0], Approx(0.75));

                        slow.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx(0.75));
                    }

                    AND_THEN("The slow backprojection yields the exact adjoint, the fast "
                             "backprojection is also exact for a very distant x-ray source")
                    {
                        sino[0] = 1;
                        fast.applyAdjoint(sino, volume);
                        // Using doubles significantly increases interpolation accuracy
                        //  For example: when using floats, points very near the border (less than
                        // ~1/500th of a pixel away from the border) are rounded to actually lie on
                        // the border. This then yields more accurate results when using floats in
                        // some of the axis-aligned test cases, despite the lower interpolation
                        // accuracy.
                        //
                        //  => different requirements for floats and doubles, looser requirements
                        // for doubles
                        if constexpr (std::is_same_v<data_t, float>)
                            REQUIRE_UNARY(isCwiseApprox(
                                volume,
                                DataContainer<data_t>(volumeDescriptor, backProjections[i / 2])));
                        else
                            REQUIRE_UNARY(isApprox(
                                volume,
                                DataContainer<data_t>(volumeDescriptor, backProjections[i / 2]),
                                static_cast<real_t>(0.005)));

                        slow.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(
                            isCwiseApprox(volume, DataContainer<data_t>(volumeDescriptor,
                                                                        backProjections[i / 2])));
                    }
                }
            }
        }

        offsetx[0] = -volSize / 2.0;
        offsetx[1] = volSize / 2.0;
        offsetx[2] = 0.0;
        offsetx[3] = 0.0;
        offsetx[4] = volSize / 2.0;
        offsetx[5] = -volSize / 2.0;

        offsety[0] = 0.0;
        offsety[1] = 0.0;
        offsety[2] = -volSize / 2.0;
        offsety[3] = volSize / 2.0;
        offsety[4] = volSize / 2.0;
        offsety[5] = -volSize / 2.0;

        // clang-format off
        backProjections[0] << 0, 0, 0, 1, 0, 0, 0, 0, 0,
                              0, 0, 0, 1, 0, 0, 0, 0, 0,
                              0, 0, 0, 1, 0, 0, 0, 0, 0;

        backProjections[1] << 0, 0, 0, 0, 0, 1, 0, 0, 0,
                              0, 0, 0, 0, 0, 1, 0, 0, 0,
                              0, 0, 0, 0, 0, 1, 0, 0, 0;

        backProjections[2] << 0, 1, 0, 0, 0, 0, 0, 0, 0,
                              0, 1, 0, 0, 0, 0, 0, 0, 0,
                              0, 1, 0, 0, 0, 0, 0, 0, 0;

        backProjections[3] << 0, 0, 0, 0, 0, 0, 0, 1, 0,
                              0, 0, 0, 0, 0, 0, 0, 1, 0,
                              0, 0, 0, 0, 0, 0, 0, 1, 0;

        backProjections[4] << 0, 0, 0, 0, 0, 0, 0, 0, 1,
                              0, 0, 0, 0, 0, 0, 0, 0, 1,
                              0, 0, 0, 0, 0, 0, 0, 0, 1;

        backProjections[5] << 1, 0, 0, 0, 0, 0, 0, 0, 0,
                              1, 0, 0, 0, 0, 0, 0, 0, 0,
                              1, 0, 0, 0, 0, 0, 0, 0, 0;
        // clang-format on

        al[0] = "left border";
        al[1] = "right border";
        al[2] = "top border";
        al[3] = "bottom border";
        al[4] = "top right edge";
        al[5] = "bottom left edge";

        for (std::size_t i = 0; i < numCases; i++) {
            WHEN("A z-axis-aligned ray runs along the corners and edges of the volume")
            {
                INFO("A z-axis-aligned ray runs along the ", al[i], " of the volume");
                // x-ray source must be very far from the volume center to make testing of the fast
                // backprojection simpler
                geom.emplace_back(SourceToCenterOfRotation{volSize * 2000}, ctr, std::move(volData),
                                  std::move(sinoData), RotationAngles3D{Gamma{0}},
                                  PrincipalPointOffset2D{0, 0},
                                  RotationOffset3D{offsetx[i], offsety[i], 0});
                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer<data_t> sino(sinoDescriptor);
                sino = 0;

                TestType fast(volumeDescriptor, sinoDescriptor);
                TestType slow(volumeDescriptor, sinoDescriptor, false);

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

                        fast.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx(1));

                        slow.apply(volume, sino);
                        REQUIRE_EQ(sino[0], Approx(1));
                    }

                    AND_THEN("The slow backprojection yields the exact adjoint, the fast "
                             "backprojection is also exact for a very distant x-ray source")
                    {
                        sino[0] = 1;
                        fast.applyAdjoint(sino, volume);

                        // Using doubles significantly increases interpolation accuracy
                        //  For example: when using floats, points very near the border (less than
                        // ~1/500th of a pixel away from the border) are rounded to actually lie on
                        // the border. This then yields more accurate results when using floats in
                        // some of the axis-aligned test cases, despite the lower interpolation
                        // accuracy.
                        //
                        //  => different requirements for floats and doubles, looser requirements
                        // for doubles
                        if constexpr (std::is_same_v<data_t, float>)
                            REQUIRE_UNARY(isCwiseApprox(
                                volume, DataContainer<data_t>(
                                            volumeDescriptor,
                                            (backProjections[i] / (i > 3 ? 4 : 2)).eval())));
                        else
                            REQUIRE_UNARY(
                                isApprox(volume,
                                         DataContainer<data_t>(
                                             volumeDescriptor,
                                             (backProjections[i] / (i > 3 ? 4 : 2)).eval()),
                                         static_cast<real_t>(0.005)));

                        slow.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(isCwiseApprox(
                            volume, DataContainer<data_t>(volumeDescriptor, backProjections[i])));
                    }
                }
            }
        }

        const data_t sqrt3d = std::sqrt(static_cast<data_t>(3));
        const data_t thirdd = static_cast<data_t>(1.0 / 3);

        Eigen::Matrix<data_t, Eigen::Dynamic, 1> backProj(volSize * volSize * volSize);

        GIVEN("A 30 degree rotate around z axis")
        {
            auto gamma = Gamma{Degree{30}};

            WHEN("A ray goes through the center of the volume")
            {
                // In this case the ray enters and exits the volume along the main direction
                geom.emplace_back(stc, ctr, std::move(volData), std::move(sinoData),
                                  RotationAngles3D{gamma});

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer<data_t> sino(sinoDescriptor);

                TestType op(volumeDescriptor, sinoDescriptor, false);

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
                        REQUIRE_EQ(sino[0], Approx(6 / sqrt3d + 2.0 / 3).epsilon(0.001));

                        sino[0] = 1;
                        // clang-format off
                            backProj << 0, 0, 0, 0         , 2 / sqrt3d - 2 * thirdd, 2 * thirdd, 0, 0, 0,
                                        0, 0, 0, 0         , 2 / sqrt3d             , 0         , 0, 0, 0,
                                        0, 0, 0, 2 * thirdd, 2 / sqrt3d - 2 * thirdd, 0         , 0, 0, 0;
                        // clang-format on

                        op.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(isCwiseApprox(
                            volume, DataContainer<data_t>(volumeDescriptor, backProj)));
                    }
                }
            }

            WHEN("A ray enters through the right border")
            {
                // In this case the ray enters through a border orthogonal to a non-main direction
                geom.emplace_back(stc, ctr, std::move(volData), std::move(sinoData),
                                  RotationAngles3D{gamma}, PrincipalPointOffset2D{0, 0},
                                  RotationOffset3D{1, 0, 0});

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer<data_t> sino(sinoDescriptor);

                TestType op(volumeDescriptor, sinoDescriptor, false);

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
                        REQUIRE_EQ(sino[0], Approx((sqrt3d + 1) * (1 - 1 / sqrt3d) + 3 - sqrt3d / 2
                                                   + 2 / sqrt3d)
                                                .epsilon(0.001));

                        sino[0] = 1;
                        // clang-format off
                        backProj << 0, 0, 0, 0, 0, ((sqrt3d + 1) / 4) * (1 - 1 / sqrt3d), 0, 0, 0,
                            0, 0, 0, 0, 0, 2 / sqrt3d + 1 - sqrt3d / 2, 0, 0, 0, 0, 0, 0, 0,
                            2 * thirdd, 2 / sqrt3d - 2 * thirdd, 0, 0, 0;
                        // clang-format on

                        op.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(isCwiseApprox(
                            volume, DataContainer<data_t>(volumeDescriptor, backProj)));
                    }
                }
            }

            WHEN("A ray exits through the left border")
            {
                // In this case the ray exit through a border orthogonal to a non-main direction
                geom.emplace_back(stc, ctr, std::move(volData), std::move(sinoData),
                                  RotationAngles3D{gamma}, PrincipalPointOffset2D{0, 0},
                                  RotationOffset3D{-1, 0, 0});

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer<data_t> sino(sinoDescriptor);

                TestType op(volumeDescriptor, sinoDescriptor, false);

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
                        REQUIRE_EQ(sino[0], Approx((sqrt3d + 1) * (1 - 1 / sqrt3d) + 3 - sqrt3d / 2
                                                   + 2 / sqrt3d)
                                                .epsilon(0.001));

                        sino[0] = 1;
                        // clang-format off
                        backProj << 0, 0, 0, 2 / sqrt3d - 2 * thirdd, 2 * thirdd, 0, 0, 0, 0, 0, 0,
                            0, 2 / sqrt3d + 1 - sqrt3d / 2, 0, 0, 0, 0, 0, 0, 0, 0,
                            ((sqrt3d + 1) / 4) * (1 - 1 / sqrt3d), 0, 0, 0, 0, 0;
                        // clang-format on

                        op.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(isCwiseApprox(
                            volume, DataContainer<data_t>(volumeDescriptor, backProj)));
                    }
                }
            }

            WHEN("A ray with an angle of 30 degrees only intersects a single voxel")
            {
                // special case - no interior voxels, entry and exit voxels are the same
                geom.emplace_back(stc, ctr, std::move(volData), std::move(sinoData),
                                  RotationAngles3D{gamma}, PrincipalPointOffset2D{0, 0},
                                  RotationOffset3D{-2, 0, 0});

                PlanarDetectorDescriptor sinoDescriptor(sinoDims, geom);
                DataContainer<data_t> sino(sinoDescriptor);

                TestType op(volumeDescriptor, sinoDescriptor, false);

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
                        REQUIRE_EQ(sino[0], Approx(sqrt3d - 1));

                        sino[0] = 1;
                        // clang-format off
                        backProj << 0, 0, 0, sqrt3d - 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0;
                        // clang-format off

                        op.applyAdjoint(sino, volume);
                        REQUIRE_UNARY(
                            isCwiseApprox(volume, DataContainer<data_t>(volumeDescriptor, backProj)));
                    }
                }
            }
        }
    }

    GIVEN("A 5x5x5 Volume")
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
        const auto stc = SourceToCenterOfRotation{20 * volSize};
        const auto ctr = CenterOfRotationToDetector{volSize};
        auto volData = VolumeData3D{Size3D{volumeDims}};
        auto sinoData = SinogramData3D{Size3D{sinoDims}};

        std::vector<Geometry> geom;

        GIVEN("Rays not intersecting the volume")
        {
            constexpr index_t numCases = 9;
            std::array<real_t, numCases> alpha = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            std::array<real_t, numCases> beta = {0.0, 0.0,      0.0,      0.0,     0.0,
                                                 0.0, pi_t / 2, pi_t / 2, pi_t / 2};
            std::array<real_t, numCases> gamma = {0.0,      0.0,      0.0,      pi_t / 2, pi_t / 2,
                                                  pi_t / 2, pi_t / 2, pi_t / 2, pi_t / 2};
            std::array<real_t, numCases> offsetx = {-volSize, 0.0,      -volSize, 0.0,     0.0,
                                                    0.0,      -volSize, 0.0,      -volSize};
            std::array<real_t, numCases> offsety = {0.0,      -volSize, -volSize, -volSize, 0.0,
                                                    -volSize, 0.0,      0.0,      0.0};
            std::array<real_t, numCases> offsetz = {0.0,      0.0, 0.0,      0.0,     -volSize,
                                                    -volSize, 0.0, -volSize, -volSize};
            std::array<std::string, numCases> neg = {"x",       "y", "x and y", "y",      "z",
                                                     "y and z", "x", "z",       "x and z"};
            std::array<std::string, numCases> ali = {"z", "z", "z", "x", "x", "x", "y", "y", "y"};

            for (std::size_t i = 0; i < numCases; i++) {
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
                    sino = 0;

                    TestType fast(volumeDescriptor, sinoDescriptor);
                    TestType slow(volumeDescriptor, sinoDescriptor, false);

                    THEN("Result of forward projection is zero")
                    {
                        fast.apply(volume, sino);
                        DataContainer<data_t> zero(sinoDescriptor);
                        zero = 0;
                        REQUIRE_UNARY(isCwiseApprox(sino, zero));

                        slow.apply(volume, sino);
                        REQUIRE_UNARY(isCwiseApprox(sino, zero));

                        AND_THEN("Result of backprojection is zero")
                        {
                            fast.applyAdjoint(sino, volume);
                            DataContainer<data_t> zero(volumeDescriptor);
                            zero = 0;
                            REQUIRE_UNARY(isCwiseApprox(volume, zero));

                            slow.applyAdjoint(sino, volume);
                            REQUIRE_UNARY(isCwiseApprox(volume, zero));
                        }
                    }
                }
            }
        }
    }
}
TEST_CASE_TEMPLATE("JosephsMethodCUDA: 3D setup with a multiple rays",TestType, JosephsMethodCUDA<float>,
                   JosephsMethodCUDA<double>)
{
    // Turn logger of
    Logger::setLevel(Logger::LogLevel::OFF);

    using data_t = decltype(return_data_t(std::declval<TestType>()));

    GIVEN("A 3x3x3 volume with 6 projection angles")
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
        const auto stc = SourceToCenterOfRotation{20 * volSize};
        const auto ctr = CenterOfRotationToDetector{volSize};
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

            TestType slow(volumeDescriptor, sinoDescriptor, false);
            TestType fast(volumeDescriptor, sinoDescriptor);

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

                slow.apply(volume, sino);
                for (index_t i = 0; i < numImgs; i++)
                    REQUIRE_EQ(sino[i], Approx(3.0));

                fast.apply(volume, sino);
                for (index_t i = 0; i < numImgs; i++)
                    REQUIRE_EQ(sino[i], Approx(3.0));

                AND_THEN("Both fast and slow backprojections yield the exact adjoint")
                {
                    Eigen::Matrix<data_t, Eigen::Dynamic, 1> cmp(volSize * volSize * volSize);

                    // clang-format off
                    cmp << 0, 0, 0, 0,  6, 0, 0, 0, 0,
                           0, 6, 0, 6, 18, 6, 0, 6, 0,
                           0, 0, 0, 0,  6, 0, 0, 0, 0;
                    // clang-format on

                    slow.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(
                        isCwiseApprox(volume, DataContainer<data_t>(volumeDescriptor, cmp)));

                    fast.applyAdjoint(sino, volume);
                    REQUIRE_UNARY(
                        isCwiseApprox(volume, DataContainer<data_t>(volumeDescriptor, cmp)));
                }
            }
        }
    }
}
