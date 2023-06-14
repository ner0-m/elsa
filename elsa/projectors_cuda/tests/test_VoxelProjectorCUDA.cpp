#include "doctest/doctest.h"

#include "VoxelProjectorCUDA.h"
#include "Geometry.h"
#include "Logger.h"
#include "VolumeDescriptor.h"
#include "PlanarDetectorDescriptor.h"
#include "testHelpers.h"

#include <array>

using namespace elsa;
using namespace elsa::geometry;
using namespace doctest;

using doctest::Approx;

/*
 * this function declaration can be used in conjunction with decltype to deduce the
 * template parameter of a templated class at compile time
 *
 * the template parameter must be a typename
 */
template <template <typename> typename T, typename data_t>
constexpr data_t return_data_t(const T<data_t>&);

TYPE_TO_STRING(BlobVoxelProjectorCUDA<float>);
// TYPE_TO_STRING(PhaseContrastBlobVoxelProjectorCUDA<float>);

#undef GIVEN
#define GIVEN(...) DOCTEST_SUBCASE((std::string("   Given: ") + std::string(__VA_ARGS__)).c_str())

TEST_SUITE_BEGIN("projectors_cuda");

TEST_CASE_TEMPLATE("VoxelProjectorCUDA: Calls to functions of super class", TestType,
                   BlobVoxelProjectorCUDA<float>, BSplineVoxelProjectorCUDA<float>,
                   BlobVoxelProjectorCUDA<double>, BSplineVoxelProjectorCUDA<double>,
                   PhaseContrastBlobVoxelProjectorCUDA<float>,
                   PhaseContrastBSplineVoxelProjectorCUDA<float>,
                   PhaseContrastBlobVoxelProjectorCUDA<double>,
                   PhaseContrastBSplineVoxelProjectorCUDA<double>)
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

TEST_CASE_TEMPLATE("BlobVoxelProjectorCUDA: Testing simple volume 3D with one detector pixel",
                   data_t, float, double)
{
    const IndexVector_t sizeDomain({{5, 5, 5}});
    const IndexVector_t sizeRange({{1, 1, 1}});

    auto stc = SourceToCenterOfRotation{100};
    auto ctr = CenterOfRotationToDetector{5};

    const RealVector_t spacing{{1, 1, 1}};
    for (int i = 0; i < 360; i += 4) {
        GIVEN("Ray of angle " + std::to_string(i))
        {
            auto domain = VolumeDescriptor(sizeDomain, spacing);
            auto x = DataContainer<data_t>(domain);
            x = 0;
            // set center voxel to 1
            x(2, 2, 2) = 1;
            std::vector<Geometry> geom;
            geom.emplace_back(stc, ctr, VolumeData3D{Size3D{sizeDomain}, Spacing3D{spacing}},
                              SinogramData3D{Size3D{sizeRange}},
                              RotationAngles3D{Gamma{static_cast<real_t>(i)}});
            auto range = PlanarDetectorDescriptor(sizeRange, geom);
            auto op = BlobVoxelProjectorCUDA<data_t>(domain, range);

            auto Ax = DataContainer<data_t>(range);
            Ax = 0;

            WHEN("projecting forward and only the center voxel is set to 1")
            {
                op.apply(x, Ax);

                const auto weight = op.blob(0);
                CAPTURE(weight);

                THEN("The detector value is the weight for distance 0")
                {
                    CAPTURE(Ax[0]);
                    CHECK_EQ(weight, Approx(Ax[0]).epsilon(0.005));
                }
            }
        }
    }
}

TEST_CASE_TEMPLATE("BlobVoxelProjectorCUDA: Testing simple volume 2D with one detector Pixel",
                   data_t, float, double)
{
    const IndexVector_t sizeDomain({{5, 5}});
    const IndexVector_t sizeRange({{1, 1}});

    auto stc = SourceToCenterOfRotation{100};
    auto ctr = CenterOfRotationToDetector{5};

    for (int i = 1; i < 4; i++) {
        real_t scaling = i / 2.0f;
        const RealVector_t spacing{{scaling, scaling}};
        GIVEN("Spacing of " + std::to_string(scaling))
        for (int i = 0; i < 360; i += 4) {
            GIVEN("Ray of angle " + std::to_string(i))
            {
                auto domain = VolumeDescriptor(sizeDomain, spacing);
                auto x = DataContainer<data_t>(domain);
                x = 0;
                std::vector<Geometry> geom;
                geom.emplace_back(stc, ctr, Degree{static_cast<real_t>(i)},
                                  VolumeData2D{Size2D{sizeDomain}, Spacing2D{spacing}},
                                  SinogramData2D{Size2D{sizeRange}});
                auto range = PlanarDetectorDescriptor(sizeRange, geom);
                auto op = BlobVoxelProjectorCUDA<data_t>(domain, range);

                auto Ax = DataContainer<data_t>(range);
                Ax = 0;

                WHEN("projecting forward and only the center voxel is set to 1")
                {
                    // set center voxel to 1
                    x(2, 2) = 1;

                    op.apply(x, Ax);

                    const auto weight = op.blob(0);
                    CAPTURE(weight);

                    THEN("The detector value is the weight for distance 0")
                    {
                        CAPTURE(Ax[0]);
                        CHECK_EQ(weight, Approx(Ax[0]));
                    }
                }
            }
        }
    }
}

TEST_CASE_TEMPLATE("BlobVoxelProjectorCUDA: Testing simple volume 2D with two detector pixels",
                   data_t, float, double)
{
    const IndexVector_t sizeDomain({{5, 5}});
    const IndexVector_t sizeRange({{2, 1}});

    auto domain = VolumeDescriptor(sizeDomain);
    auto x = DataContainer<data_t>(domain);
    x = 0;

    auto stc = SourceToCenterOfRotation{100};
    auto ctr = CenterOfRotationToDetector{5};
    auto volData = VolumeData2D{Size2D{sizeDomain}};
    auto sinoData = SinogramData2D{Size2D{sizeRange}};

    for (int i = 0; i < 360; i += 4) {
        GIVEN("Ray of angle " + std::to_string(i))
        {
            std::vector<Geometry> geom;
            geom.emplace_back(stc, ctr, Degree{static_cast<real_t>(i)},
                              VolumeData2D{Size2D{sizeDomain}}, SinogramData2D{Size2D{sizeRange}});
            auto range = PlanarDetectorDescriptor(sizeRange, geom);
            auto op = BlobVoxelProjectorCUDA<data_t>(domain, range);

            auto Ax = DataContainer<data_t>(range);
            Ax = 0;

            WHEN("only the center voxel is set to 1")
            {
                // set center voxel to 1
                x(2, 2) = 1;

                op.apply(x, Ax);

                const auto weight_dist_half = op.blob(0.4761878);
                CAPTURE(weight_dist_half);

                THEN("Detector values are symmetric")
                {
                    CAPTURE(Ax[0]);
                    CAPTURE(Ax[1]);
                    CHECK_EQ(weight_dist_half, Approx(Ax[0]).epsilon(0.01));
                    CHECK_EQ(weight_dist_half, Approx(Ax[1]).epsilon(0.01));
                }
            }
        }
    }
}

TEST_CASE_TEMPLATE("BlobVoxelProjectorCUDA: Testing simple volume 2D with three detector pixels",
                   data_t, float, double)
{
    const IndexVector_t sizeDomain({{5, 5}});
    const IndexVector_t sizeRange({{3, 1}});

    auto domain = VolumeDescriptor(sizeDomain);
    auto x = DataContainer<data_t>(domain);
    x = 0;

    auto stc = SourceToCenterOfRotation{100};
    auto ctr = CenterOfRotationToDetector{5};
    auto volData = VolumeData2D{Size2D{sizeDomain}};
    auto sinoData = SinogramData2D{Size2D{sizeRange}};

    for (int i = 0; i < 360; i += 5) {
        GIVEN("Ray of angle " + std::to_string(i))
        {
            std::vector<Geometry> geom;
            geom.emplace_back(stc, ctr, Degree{static_cast<real_t>(i)},
                              VolumeData2D{Size2D{sizeDomain}}, SinogramData2D{Size2D{sizeRange}});
            auto range = PlanarDetectorDescriptor(sizeRange, geom);
            auto op = BlobVoxelProjectorCUDA<data_t>(domain, range);

            auto Ax = DataContainer<data_t>(range);
            Ax = 0;

            WHEN("only the center voxel is set to 1")
            {
                // set center voxel to 1
                x(2, 2) = 1;

                op.apply(x, Ax);

                const auto weight_dist0 = op.blob(0);
                const auto weight_dist1 = op.blob(0.95233786);
                CAPTURE(weight_dist0);
                CAPTURE(weight_dist1);

                THEN("the center detector pixel is correct")
                {
                    CAPTURE(Ax[1]);
                    CHECK_EQ(weight_dist0, Approx(Ax[1]));
                }

                THEN("the outer detector pixels are the same")
                {
                    CAPTURE(Ax[0]);
                    CAPTURE(Ax[2]);
                    CHECK_EQ(weight_dist1, Approx(Ax[0]).epsilon(0.01));
                    CHECK_EQ(weight_dist1, Approx(Ax[2]).epsilon(0.01));
                }
            }
        }
    }
}

TEST_CASE_TEMPLATE("BlobVoxelProjectorCUDA: Test single detector pixel", data_t, float, double)
{
    const IndexVector_t sizeDomain({{5, 5}});
    const IndexVector_t sizeRange({{1, 1}});

    auto domain = VolumeDescriptor(sizeDomain);
    auto x = DataContainer<data_t>(domain);
    x = 0;

    auto stc = SourceToCenterOfRotation{100};
    auto ctr = CenterOfRotationToDetector{5};
    auto volData = VolumeData2D{Size2D{sizeDomain}};
    auto sinoData = SinogramData2D{Size2D{sizeRange}};

    GIVEN("a single detector of size 1, at 0 degree")
    {
        std::vector<Geometry> geom;
        geom.emplace_back(stc, ctr, Degree{0}, VolumeData2D{Size2D{sizeDomain}},
                          SinogramData2D{Size2D{sizeRange}});

        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BlobVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;

        WHEN("Setting only the voxels directly on the ray to 1")
        {
            // set all voxels on the ray to 1
            x(2, 0) = 1;
            x(2, 1) = 1;
            x(2, 2) = 1;
            x(2, 3) = 1;
            x(2, 4) = 1;

            op.apply(x, Ax);

            const auto weight = op.blob(0);
            CAPTURE(weight);

            THEN("Each detector value is equal to 5 * the weight of distance 0")
            {
                CHECK_EQ(Ax[0], Approx(5 * weight));
            }
        }

        WHEN("Setting only the voxels direct neighbours above the ray to 1")
        {
            // set all voxels above the ray to 1, i.e the visited neighbours
            x(3, 0) = 1;
            x(3, 1) = 1;
            x(3, 2) = 1;
            x(3, 3) = 1;
            x(3, 4) = 1;

            op.apply(x, Ax);

            THEN("The detector value is equal to the weight at the scaled distances")
            {
                data_t total_weight = 0;
                for (real_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{3.f, slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.blob(detectorCoord / scaling);
                }

                CHECK_EQ(Ax[0], Approx(total_weight));
            }
        }

        WHEN("Setting only the voxels direct neighbours below the ray to 1")
        {
            // set all voxels above the ray to 1, i.e the visited neighbours
            x(1, 0) = 1;
            x(1, 1) = 1;
            x(1, 2) = 1;
            x(1, 3) = 1;
            x(1, 4) = 1;

            op.apply(x, Ax);

            THEN("The detector value is equal to the weight at the scaled distances")
            {
                data_t total_weight = 0;
                for (real_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{1.f, slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.blob(detectorCoord / scaling);
                }

                CHECK_EQ(Ax[0], Approx(total_weight));
            }
        }
    }

    GIVEN("a single detector of size 1, at 45 degree")
    {
        std::vector<Geometry> geom;
        geom.emplace_back(stc, ctr, Degree{45}, VolumeData2D{Size2D{sizeDomain}},
                          SinogramData2D{Size2D{sizeRange}});

        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BlobVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;

        WHEN("Setting only the voxels directly on the ray to 1")
        {
            // set all voxels on the ray to 1
            x(0, 0) = 1;
            x(1, 1) = 1;
            x(2, 2) = 1;
            x(3, 3) = 1;
            x(4, 4) = 1;

            op.apply(x, Ax);

            const auto weight = op.blob(0);
            CAPTURE(weight);

            THEN("Each detector value is equal to 5 * the weight of distance 0")
            {
                CHECK_EQ(Ax[0], Approx(5 * weight));
            }
        }

        WHEN("Setting only the voxels directly above the ray to 1")
        {
            // set all voxels on the ray to 1
            x(0, 1) = 1;
            x(1, 2) = 1;
            x(2, 3) = 1;
            x(3, 4) = 1;

            op.apply(x, Ax);

            THEN("The detector value is equal to the weight at the scaled distances")
            {
                data_t total_weight = 0;
                for (real_t slice : {0, 1, 2, 3}) {
                    RealVector_t voxCoord{{slice, slice + 1}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.blob(detectorCoord / scaling);
                }

                CHECK_EQ(Ax[0], Approx(total_weight));
            }
        }

        WHEN("Setting only the voxels directly below the ray to 1")
        {
            // set all voxels on the ray to 1
            x(1, 0) = 1;
            x(2, 1) = 1;
            x(3, 2) = 1;
            x(4, 3) = 1;

            op.apply(x, Ax);

            THEN("The detector value is equal to the weight at the scaled distances")
            {
                data_t total_weight = 0;
                for (real_t slice : {0, 1, 2, 3}) {
                    RealVector_t voxCoord{{slice + 1, slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.blob(detectorCoord / scaling);
                }

                CHECK_EQ(Ax[0], Approx(total_weight));
            }
        }

        WHEN("Setting only the voxels two steps above the ray")
        {
            // set all voxels on the ray to 1
            x(0, 2) = 1;
            x(1, 3) = 1;
            x(2, 4) = 1;

            op.apply(x, Ax);

            THEN("The detector value is equal to the weight at the scaled distances")
            {
                data_t total_weight = 0;
                for (real_t slice : {0, 1, 2}) {
                    RealVector_t voxCoord{{slice, slice + 2}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.blob(detectorCoord / scaling);
                }

                CHECK_EQ(Ax[0], Approx(total_weight));
            }
        }

        WHEN("Setting only the voxels two steps below the ray")
        {
            // set all voxels on the ray to 1
            x(2, 0) = 1;
            x(3, 1) = 1;
            x(4, 2) = 1;

            op.apply(x, Ax);

            THEN("The detector value is equal to the weight at the scaled distances")
            {
                data_t total_weight = 0;
                for (real_t slice : {0, 1, 2}) {
                    RealVector_t voxCoord{{slice + 2, slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.blob(detectorCoord / scaling);
                }

                CHECK_EQ(Ax[0], Approx(total_weight));
            }
        }
    }

    GIVEN("a single detector of size 1, at 90 degree")
    {
        std::vector<Geometry> geom;
        geom.emplace_back(stc, ctr, Degree{90}, VolumeData2D{Size2D{sizeDomain}},
                          SinogramData2D{Size2D{sizeRange}});

        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BlobVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;

        WHEN("Applying the BlobVoxelProjectorCUDA<data_t> to the volume")
        {
            x = 0;
            // set all voxels on the ray to 1
            x(0, 2) = 1;
            x(1, 2) = 1;
            x(2, 2) = 1;
            x(3, 2) = 1;
            x(4, 2) = 1;

            op.apply(x, Ax);

            const auto weight = op.blob(0);
            CAPTURE(weight);

            THEN("Each detector value is equal to 5 * the weight of distance 0")
            {
                CHECK_EQ(Ax[0], Approx(5 * weight));
            }
        }

        WHEN("Setting only the voxels direct neighbours above the ray to 1")
        {
            // set all voxels above the ray to 1, i.e the visited neighbours
            x(0, 3) = 1;
            x(1, 3) = 1;
            x(2, 3) = 1;
            x(3, 3) = 1;
            x(4, 3) = 1;

            op.apply(x, Ax);

            THEN("The detector value is equal to the weight at the scaled distances")
            {
                data_t total_weight = 0;
                for (real_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{slice, 3.f}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.blob(detectorCoord / scaling);
                }

                CHECK_EQ(Ax[0], Approx(total_weight));
            }
        }

        WHEN("Setting only the voxels direct neighbours below the ray to 1")
        {
            // set all voxels above the ray to 1, i.e the visited neighbours
            x(0, 1) = 1;
            x(1, 1) = 1;
            x(2, 1) = 1;
            x(3, 1) = 1;
            x(4, 1) = 1;

            op.apply(x, Ax);

            THEN("The detector value is equal to the weight at the scaled distances")
            {
                data_t total_weight = 0;
                for (real_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{slice, 1.f}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.blob(detectorCoord / scaling);
                }

                CHECK_EQ(Ax[0], Approx(total_weight));
            }
        }
    }

    GIVEN("a single detector of size 1, at 135 degree")
    {
        std::vector<Geometry> geom;
        geom.emplace_back(stc, ctr, Degree{135}, VolumeData2D{Size2D{sizeDomain}},
                          SinogramData2D{Size2D{sizeRange}});

        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BlobVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;

        WHEN("Applying the BlobVoxelProjectorCUDA<data_t> to the volume")
        {
            // set all voxels on the ray to 1
            x(0, 4) = 1;
            x(1, 3) = 1;
            x(2, 2) = 1;
            x(3, 1) = 1;
            x(4, 0) = 1;

            op.apply(x, Ax);

            const auto weight = op.blob(0);
            CAPTURE(weight);

            THEN("Each detector value is equal to 5 * the weight of distance 0")
            {
                CHECK_EQ(Ax[0], Approx(5 * weight));
            }
        }

        WHEN("Setting only the voxels directly above on the ray to 1")
        {
            // set all voxels on the ray to 1
            x(1, 4) = 1;
            x(2, 3) = 1;
            x(3, 2) = 1;
            x(4, 1) = 1;

            op.apply(x, Ax);

            THEN("The detector value is equal to the weight at the scaled distances")
            {
                data_t total_weight = 0;
                for (real_t slice : {1, 2, 3, 4}) {
                    RealVector_t voxCoord{{slice, 5 - slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.blob(detectorCoord / scaling);
                }

                CHECK_EQ(Ax[0], Approx(total_weight));
            }
        }

        WHEN("Setting only the voxels directly above on the ray to 1")
        {
            // set all voxels on the ray to 1
            x(0, 3) = 1;
            x(1, 2) = 1;
            x(2, 1) = 1;
            x(3, 0) = 1;

            op.apply(x, Ax);

            THEN("The detector value is equal to the weight at the scaled distances")
            {
                data_t total_weight = 0;
                for (real_t slice : {0, 1, 2, 3}) {
                    RealVector_t voxCoord{{slice, 3 - slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.blob(detectorCoord / scaling);
                }

                CHECK_EQ(Ax[0], Approx(total_weight));
            }
        }

        WHEN("Setting only the voxels two above on the ray")
        {
            // set all voxels on the ray to 1
            x(2, 4) = 1;
            x(3, 3) = 1;
            x(4, 2) = 1;

            op.apply(x, Ax);

            THEN("The detector value is equal to the weight at the scaled distances")
            {
                data_t total_weight = 0;
                for (real_t slice : {2, 3, 4}) {
                    RealVector_t voxCoord{{slice, 6 - slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.blob(detectorCoord / scaling);
                }

                CHECK_EQ(Ax[0], Approx(total_weight));
            }
        }

        WHEN("Setting only the voxels directly above on the ray to 1")
        {
            // set all voxels on the ray to 1
            x(0, 2) = 1;
            x(1, 1) = 1;
            x(2, 0) = 1;

            op.apply(x, Ax);

            THEN("The detector value is equal to the weight at the scaled distances")
            {
                data_t total_weight = 0;
                for (real_t slice : {0, 1, 2}) {
                    RealVector_t voxCoord{{slice, 2 - slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.blob(detectorCoord / scaling);
                }

                CHECK_EQ(Ax[0], Approx(total_weight));
            }
        }
    }

    GIVEN("a single detector of size 1, at 180 degree")
    {
        std::vector<Geometry> geom;
        geom.emplace_back(stc, ctr, Degree{180}, VolumeData2D{Size2D{sizeDomain}},
                          SinogramData2D{Size2D{sizeRange}});

        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BlobVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;

        WHEN("Applying the BlobVoxelProjectorCUDA to the volume")
        {
            x = 0;
            // set all voxels on the ray to 1
            x(2, 0) = 1;
            x(2, 1) = 1;
            x(2, 2) = 1;
            x(2, 3) = 1;
            x(2, 4) = 1;

            op.apply(x, Ax);

            const auto weight = op.blob(0);
            CAPTURE(weight);

            THEN("Each detector value is equal to 5 * the weight of distance 0")
            {
                CHECK_EQ(Ax[0], Approx(5 * weight));
            }
        }

        WHEN("Setting only the voxels direct neighbours above the ray to 1")
        {
            // set all voxels above the ray to 1, i.e the visited neighbours
            x(3, 0) = 1;
            x(3, 1) = 1;
            x(3, 2) = 1;
            x(3, 3) = 1;
            x(3, 4) = 1;

            op.apply(x, Ax);

            THEN("The detector value is equal to the weight at the scaled distances")
            {
                data_t total_weight = 0;
                for (real_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{3.f, slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.blob(detectorCoord / scaling);
                }

                CHECK_EQ(Ax[0], Approx(total_weight));
            }
        }

        WHEN("Setting only the voxels direct neighbours below the ray to 1")
        {
            // set all voxels above the ray to 1, i.e the visited neighbours
            x(1, 0) = 1;
            x(1, 1) = 1;
            x(1, 2) = 1;
            x(1, 3) = 1;
            x(1, 4) = 1;

            op.apply(x, Ax);

            THEN("The detector value is equal to the weight at the scaled distances")
            {
                data_t total_weight = 0;
                for (real_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{1.f, slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.blob(detectorCoord / scaling);
                }

                CHECK_EQ(Ax[0], Approx(total_weight));
            }
        }
    }

    GIVEN("a single detector of size 1, at 225 degree")
    {
        std::vector<Geometry> geom;
        geom.emplace_back(stc, ctr, Degree{225}, VolumeData2D{Size2D{sizeDomain}},
                          SinogramData2D{Size2D{sizeRange}});

        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BlobVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;

        WHEN("Applying the BlobVoxelProjectorCUDA to the volume")
        {
            // set all voxels on the ray to 1
            x(0, 0) = 1;
            x(1, 1) = 1;
            x(2, 2) = 1;
            x(3, 3) = 1;
            x(4, 4) = 1;

            op.apply(x, Ax);

            const auto weight = op.blob(0);
            CAPTURE(weight);

            THEN("Each detector value is equal to 5 * the weight of distance 0")
            {
                CHECK_EQ(Ax[0], Approx(5 * weight));
            }
        }
    }

    GIVEN("a single detector of size 1, at 270 degree")
    {
        std::vector<Geometry> geom;
        geom.emplace_back(stc, ctr, Degree{270}, VolumeData2D{Size2D{sizeDomain}},
                          SinogramData2D{Size2D{sizeRange}});

        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BlobVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;

        WHEN("Applying the BlobVoxelProjectorCUDA to the volume")
        {
            x = 0;
            // set all voxels on the ray to 1
            x(0, 2) = 1;
            x(1, 2) = 1;
            x(2, 2) = 1;
            x(3, 2) = 1;
            x(4, 2) = 1;

            op.apply(x, Ax);

            const auto weight = op.blob(0);
            CAPTURE(weight);

            THEN("Each detector value is equal to 5 * the weight of distance 0")
            {
                CHECK_EQ(Ax[0], Approx(5 * weight));
            }
        }
    }

    GIVEN("a single detector of size 1, at 315 degree")
    {
        std::vector<Geometry> geom;
        geom.emplace_back(stc, ctr, Degree{315}, VolumeData2D{Size2D{sizeDomain}},
                          SinogramData2D{Size2D{sizeRange}});

        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BlobVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;

        WHEN("Applying the BlobVoxelProjectorCUDA to the volume")
        {
            // set all voxels on the ray to 1
            x(0, 4) = 1;
            x(1, 3) = 1;
            x(2, 2) = 1;
            x(3, 1) = 1;
            x(4, 0) = 1;

            op.apply(x, Ax);

            const auto weight = op.blob(0);
            CAPTURE(weight);

            THEN("Each detector value is equal to 5 * the weight of distance 0")
            {
                CHECK_EQ(Ax[0], Approx(5 * weight));
            }
        }
    }
}

TEST_CASE_TEMPLATE("BlobVoxelProjectorCUDA: Test backward projection", data_t, float, double)
{
    const IndexVector_t sizeDomain({{5, 5}});
    const IndexVector_t sizeRange({{1, 1}});

    auto domain = VolumeDescriptor(sizeDomain);
    auto x = DataContainer<data_t>(domain);

    auto stc = SourceToCenterOfRotation{100};
    auto ctr = CenterOfRotationToDetector{5};
    auto volData = VolumeData2D{Size2D{sizeDomain}};
    auto sinoData = SinogramData2D{Size2D{sizeRange}};

    GIVEN("a single detector of size 1, at 0 degree")
    {
        std::vector<Geometry> geom;
        geom.emplace_back(stc, ctr, Degree{0}, VolumeData2D{Size2D{sizeDomain}},
                          SinogramData2D{Size2D{sizeRange}});

        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BlobVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;
        x = 0;

        WHEN("Backward projecting the operator")
        {
            Ax[0] = 1;

            op.applyAdjoint(Ax, x);

            THEN("The main direction of the ray is equal to the weight")
            {
                const auto weight = op.blob(0);
                CAPTURE(weight);

                CHECK_EQ(x(2, 0), Approx(weight));
                CHECK_EQ(x(2, 1), Approx(weight));
                CHECK_EQ(x(2, 2), Approx(weight));
                CHECK_EQ(x(2, 3), Approx(weight));
                CHECK_EQ(x(2, 4), Approx(weight));
            }

            THEN("The pixel 1 above the main direction of the ray have the projected weight")
            {
                for (index_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{3.f, static_cast<float>(slice)}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.blob(detectorCoord / scaling);
                    CHECK_EQ(x(3, slice), Approx(weight));
                }
            }
            THEN("The pixel 1 below the main direction of the ray have the projected weight")
            {
                for (index_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{1.f, static_cast<float>(slice)}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.blob(detectorCoord / scaling);
                    CHECK_EQ(x(1, slice), Approx(weight));
                }
            }
        }
    }

    GIVEN("a single detector of size 1, at 180 degree")
    {
        std::vector<Geometry> geom;
        geom.emplace_back(stc, ctr, Degree{180}, VolumeData2D{Size2D{sizeDomain}},
                          SinogramData2D{Size2D{sizeRange}});

        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BlobVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;
        x = 0;

        WHEN("Backward projecting the operator")
        {
            Ax[0] = 1;

            op.applyAdjoint(Ax, x);

            THEN("The main direction of the ray is equal to the weight")
            {
                const auto weight = op.blob(0);
                CAPTURE(weight);

                CHECK_EQ(x(2, 0), Approx(weight));
                CHECK_EQ(x(2, 1), Approx(weight));
                CHECK_EQ(x(2, 2), Approx(weight));
                CHECK_EQ(x(2, 3), Approx(weight));
                CHECK_EQ(x(2, 4), Approx(weight));
            }

            THEN("The pixel 1 above the main direction of the ray have the projected weight")
            {
                for (index_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{3.f, static_cast<float>(slice)}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.blob(detectorCoord / scaling);
                    CHECK_EQ(x(3, slice), Approx(weight));
                }
            }
            THEN("The pixel 1 below the main direction of the ray have the projected weight")
            {
                for (index_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{1.f, static_cast<float>(slice)}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.blob(detectorCoord / scaling);
                    CHECK_EQ(x(1, slice), Approx(weight));
                }
            }
        }
    }

    GIVEN("a single detector of size 1, at 90 degree")
    {
        std::vector<Geometry> geom;
        geom.emplace_back(stc, ctr, Degree{90}, VolumeData2D{Size2D{sizeDomain}},
                          SinogramData2D{Size2D{sizeRange}});

        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BlobVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;
        x = 0;

        WHEN("Backward projecting the operator")
        {
            Ax[0] = 1;

            op.applyAdjoint(Ax, x);

            THEN("The main direction of the ray is equal to the weight")
            {
                const auto weight = op.blob(0);
                CAPTURE(weight);

                CHECK_EQ(x(0, 2), Approx(weight));
                CHECK_EQ(x(1, 2), Approx(weight));
                CHECK_EQ(x(2, 2), Approx(weight));
                CHECK_EQ(x(3, 2), Approx(weight));
                CHECK_EQ(x(4, 2), Approx(weight));
            }

            THEN("The pixel 1 above the main direction of the ray have the projected weight")
            {
                for (index_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{static_cast<float>(slice), 3.f}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.blob(detectorCoord / scaling);
                    CHECK_EQ(x(slice, 3), Approx(weight));
                }
            }
            THEN("The pixel 1 below the main direction of the ray have the projected weight")
            {
                for (index_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{static_cast<float>(slice), 1.f}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.blob(detectorCoord / scaling);
                    CHECK_EQ(x(slice, 1), Approx(weight));
                }
            }
        }
    }

    GIVEN("a single detector of size 1, at 270 degree")
    {
        std::vector<Geometry> geom;
        geom.emplace_back(stc, ctr, Degree{270}, VolumeData2D{Size2D{sizeDomain}},
                          SinogramData2D{Size2D{sizeRange}});

        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BlobVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;
        x = 0;

        WHEN("Backward projecting the operator")
        {
            Ax[0] = 1;

            op.applyAdjoint(Ax, x);

            THEN("The main direction of the ray is equal to the weight")
            {
                const auto weight = op.blob(0);
                CAPTURE(weight);

                CHECK_EQ(x(0, 2), Approx(weight));
                CHECK_EQ(x(1, 2), Approx(weight));
                CHECK_EQ(x(2, 2), Approx(weight));
                CHECK_EQ(x(3, 2), Approx(weight));
                CHECK_EQ(x(4, 2), Approx(weight));
            }
            THEN("The pixel 1 above the main direction of the ray have the projected weight")
            {
                for (index_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{static_cast<float>(slice), 3.f}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.blob(detectorCoord / scaling);
                    CHECK_EQ(x(slice, 3), Approx(weight));
                }
            }
            THEN("The pixel 1 below the main direction of the ray have the projected weight")
            {
                for (index_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{static_cast<float>(slice), 1.f}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.blob(detectorCoord / scaling);
                    CHECK_EQ(x(slice, 1), Approx(weight));
                }
            }
        }
    }

    GIVEN("a single detector of size 1, at 45 degree")
    {
        std::vector<Geometry> geom;
        geom.emplace_back(stc, ctr, Degree{45}, VolumeData2D{Size2D{sizeDomain}},
                          SinogramData2D{Size2D{sizeRange}});

        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BlobVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;
        x = 0;

        WHEN("Backward projecting the operator")
        {
            Ax[0] = 1;

            op.applyAdjoint(Ax, x);

            THEN("The main direction of the ray is equal to the weight")
            {
                const auto weight = op.blob(0);
                CAPTURE(weight);

                CHECK_EQ(x(0, 0), Approx(weight));
                CHECK_EQ(x(1, 1), Approx(weight));
                CHECK_EQ(x(2, 2), Approx(weight));
                CHECK_EQ(x(3, 3), Approx(weight));
                CHECK_EQ(x(4, 4), Approx(weight));
            }

            THEN("The pixel 1 above the main direction of the ray have the projected weight")
            {
                for (index_t slice : {0, 1, 2, 3}) {
                    auto sliceF = static_cast<float>(slice);
                    RealVector_t voxCoord{{sliceF, sliceF + 1}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.blob(detectorCoord / scaling);
                    CHECK_EQ(x(slice, slice + 1), Approx(weight));
                }
            }

            THEN("The pixel 1 below the main direction of the ray have the projected weight")
            {
                for (index_t slice : {0, 1, 2, 3}) {
                    auto sliceF = static_cast<float>(slice);
                    RealVector_t voxCoord{{sliceF + 1, sliceF}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.blob(detectorCoord / scaling);
                    CHECK_EQ(x(slice + 1, slice), Approx(weight));
                }
            }

            THEN("The pixel 2 above the main direction of the ray have the projected weight")
            {
                for (index_t slice : {0, 1, 2}) {
                    auto sliceF = static_cast<float>(slice);
                    RealVector_t voxCoord{{sliceF, 2 - sliceF}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.blob(detectorCoord / scaling);
                    CHECK_EQ(x(slice, 2 - slice), Approx(weight));
                }
            }

            THEN("The pixel 2 above the main direction of the ray have the projected weight")
            {
                for (index_t slice : {0, 1, 2}) {
                    auto sliceF = static_cast<float>(slice);
                    RealVector_t voxCoord{{2 - sliceF, sliceF}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.blob(detectorCoord / scaling);
                    CHECK_EQ(x(2 - slice, slice), Approx(weight));
                }
            }
        }
    }
}

//***************************************************************************************    BSPLINE
TEST_CASE_TEMPLATE("BSplineVoxelProjectorCUDA: Testing simple volume 3D with one detector pixel",
                   data_t, float, double)
{
    const IndexVector_t sizeDomain({{5, 5, 5}});
    const IndexVector_t sizeRange({{1, 1, 1}});

    auto stc = SourceToCenterOfRotation{100};
    auto ctr = CenterOfRotationToDetector{5};

    const RealVector_t spacing{{1, 1, 1}};
    for (int i = 0; i < 360; i += 4) {
        GIVEN("Ray of angle " + std::to_string(i))
        {
            auto domain = VolumeDescriptor(sizeDomain, spacing);
            auto x = DataContainer<data_t>(domain);
            x = 0;
            // set center voxel to 1
            x(2, 2, 2) = 1;
            std::vector<Geometry> geom;
            geom.emplace_back(stc, ctr, VolumeData3D{Size3D{sizeDomain}, Spacing3D{spacing}},
                              SinogramData3D{Size3D{sizeRange}},
                              RotationAngles3D{Gamma{static_cast<real_t>(i)}});
            auto range = PlanarDetectorDescriptor(sizeRange, geom);
            auto op = BSplineVoxelProjectorCUDA<data_t>(domain, range);

            auto Ax = DataContainer<data_t>(range);
            Ax = 0;

            WHEN("projecting forward and only the center voxel is set to 1")
            {
                op.apply(x, Ax);

                const auto weight = op.bspline(0);
                CAPTURE(weight);

                THEN("The detector value is the weight for distance 0")
                {
                    CAPTURE(Ax[0]);
                    CHECK_EQ(weight, Approx(Ax[0]).epsilon(0.005));
                }
            }
        }
    }
}

TEST_CASE_TEMPLATE("BSplineVoxelProjectorCUDA: Testing simple volume 2D with one detector Pixel",
                   data_t, float, double)
{
    const IndexVector_t sizeDomain({{5, 5}});
    const IndexVector_t sizeRange({{1, 1}});

    auto stc = SourceToCenterOfRotation{100};
    auto ctr = CenterOfRotationToDetector{5};

    for (int i = 1; i < 4; i++) {
        real_t scaling = i / 2.0f;
        const RealVector_t spacing{{scaling, scaling}};
        GIVEN("Spacing of " + std::to_string(scaling))
        for (int i = 0; i < 360; i += 4) {
            GIVEN("Ray of angle " + std::to_string(i))
            {
                auto domain = VolumeDescriptor(sizeDomain, spacing);
                auto x = DataContainer<data_t>(domain);
                x = 0;
                std::vector<Geometry> geom;
                geom.emplace_back(stc, ctr, Degree{static_cast<real_t>(i)},
                                  VolumeData2D{Size2D{sizeDomain}, Spacing2D{spacing}},
                                  SinogramData2D{Size2D{sizeRange}});
                auto range = PlanarDetectorDescriptor(sizeRange, geom);
                auto op = BSplineVoxelProjectorCUDA<data_t>(domain, range);

                auto Ax = DataContainer<data_t>(range);
                Ax = 0;

                WHEN("projecting forward and only the center voxel is set to 1")
                {
                    // set center voxel to 1
                    x(2, 2) = 1;

                    op.apply(x, Ax);

                    const auto weight = op.bspline(0);
                    CAPTURE(weight);

                    THEN("The detector value is the weight for distance 0")
                    {
                        CAPTURE(Ax[0]);
                        CHECK_EQ(weight, Approx(Ax[0]));
                    }
                }
            }
        }
    }
}

TEST_CASE_TEMPLATE("BSplineVoxelProjectorCUDA: Testing simple volume 2D with two detector pixels",
                   data_t, float, double)
{
    const IndexVector_t sizeDomain({{5, 5}});
    const IndexVector_t sizeRange({{2, 1}});

    auto domain = VolumeDescriptor(sizeDomain);
    auto x = DataContainer<data_t>(domain);
    x = 0;

    auto stc = SourceToCenterOfRotation{100};
    auto ctr = CenterOfRotationToDetector{5};
    auto volData = VolumeData2D{Size2D{sizeDomain}};
    auto sinoData = SinogramData2D{Size2D{sizeRange}};

    for (int i = 0; i < 360; i += 4) {
        GIVEN("Ray of angle " + std::to_string(i))
        {
            std::vector<Geometry> geom;
            geom.emplace_back(stc, ctr, Degree{static_cast<real_t>(i)},
                              VolumeData2D{Size2D{sizeDomain}}, SinogramData2D{Size2D{sizeRange}});
            auto range = PlanarDetectorDescriptor(sizeRange, geom);
            auto op = BSplineVoxelProjectorCUDA<data_t>(domain, range);

            auto Ax = DataContainer<data_t>(range);
            Ax = 0;

            WHEN("only the center voxel is set to 1")
            {
                // set center voxel to 1
                x(2, 2) = 1;

                op.apply(x, Ax);

                const auto weight_dist_half = op.bspline(0.4761878);
                CAPTURE(weight_dist_half);

                THEN("Detector values are symmetric")
                {
                    CAPTURE(Ax[0]);
                    CAPTURE(Ax[1]);
                    CHECK_EQ(weight_dist_half, Approx(Ax[0]).epsilon(0.01));
                    CHECK_EQ(weight_dist_half, Approx(Ax[1]).epsilon(0.01));
                }
            }
        }
    }
}

TEST_CASE_TEMPLATE("BSplineVoxelProjectorCUDA: Testing simple volume 2D with three detector pixels",
                   data_t, float, double)
{
    const IndexVector_t sizeDomain({{5, 5}});
    const IndexVector_t sizeRange({{3, 1}});

    auto domain = VolumeDescriptor(sizeDomain);
    auto x = DataContainer<data_t>(domain);
    x = 0;

    auto stc = SourceToCenterOfRotation{100};
    auto ctr = CenterOfRotationToDetector{5};
    auto volData = VolumeData2D{Size2D{sizeDomain}};
    auto sinoData = SinogramData2D{Size2D{sizeRange}};

    for (int i = 0; i < 360; i += 5) {
        GIVEN("Ray of angle " + std::to_string(i))
        {
            std::vector<Geometry> geom;
            geom.emplace_back(stc, ctr, Degree{static_cast<real_t>(i)},
                              VolumeData2D{Size2D{sizeDomain}}, SinogramData2D{Size2D{sizeRange}});
            auto range = PlanarDetectorDescriptor(sizeRange, geom);
            auto op = BSplineVoxelProjectorCUDA<data_t>(domain, range);

            auto Ax = DataContainer<data_t>(range);
            Ax = 0;

            WHEN("only the center voxel is set to 1")
            {
                // set center voxel to 1
                x(2, 2) = 1;

                op.apply(x, Ax);

                const auto weight_dist0 = op.bspline(0);
                const auto weight_dist1 = op.bspline(0.95233786);
                CAPTURE(weight_dist0);
                CAPTURE(weight_dist1);

                THEN("the center detector pixel is correct")
                {
                    CAPTURE(Ax[1]);
                    CHECK_EQ(weight_dist0, Approx(Ax[1]));
                }

                THEN("the outer detector pixels are the same")
                {
                    CAPTURE(Ax[0]);
                    CAPTURE(Ax[2]);
                    CHECK_EQ(weight_dist1, Approx(Ax[0]).epsilon(0.01));
                    CHECK_EQ(weight_dist1, Approx(Ax[2]).epsilon(0.01));
                }
            }
        }
    }
}

TEST_CASE_TEMPLATE("BSplineVoxelProjectorCUDA: Test single detector pixel", data_t, float, double)
{
    const IndexVector_t sizeDomain({{5, 5}});
    const IndexVector_t sizeRange({{1, 1}});

    auto domain = VolumeDescriptor(sizeDomain);
    auto x = DataContainer<data_t>(domain);
    x = 0;

    auto stc = SourceToCenterOfRotation{100};
    auto ctr = CenterOfRotationToDetector{5};
    auto volData = VolumeData2D{Size2D{sizeDomain}};
    auto sinoData = SinogramData2D{Size2D{sizeRange}};

    GIVEN("a single detector of size 1, at 0 degree")
    {
        std::vector<Geometry> geom;
        geom.emplace_back(stc, ctr, Degree{0}, VolumeData2D{Size2D{sizeDomain}},
                          SinogramData2D{Size2D{sizeRange}});

        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BSplineVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;

        WHEN("Setting only the voxels directly on the ray to 1")
        {
            // set all voxels on the ray to 1
            x(2, 0) = 1;
            x(2, 1) = 1;
            x(2, 2) = 1;
            x(2, 3) = 1;
            x(2, 4) = 1;

            op.apply(x, Ax);

            const auto weight = op.bspline(0);
            CAPTURE(weight);

            THEN("Each detector value is equal to 5 * the weight of distance 0")
            {
                CHECK_EQ(Ax[0], Approx(5 * weight));
            }
        }

        WHEN("Setting only the voxels direct neighbours above the ray to 1")
        {
            // set all voxels above the ray to 1, i.e the visited neighbours
            x(3, 0) = 1;
            x(3, 1) = 1;
            x(3, 2) = 1;
            x(3, 3) = 1;
            x(3, 4) = 1;

            op.apply(x, Ax);

            THEN("The detector value is equal to the weight at the scaled distances")
            {
                data_t total_weight = 0;
                for (real_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{3.f, slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.bspline(detectorCoord / scaling);
                }

                CHECK_EQ(Ax[0], Approx(total_weight));
            }
        }

        WHEN("Setting only the voxels direct neighbours below the ray to 1")
        {
            // set all voxels above the ray to 1, i.e the visited neighbours
            x(1, 0) = 1;
            x(1, 1) = 1;
            x(1, 2) = 1;
            x(1, 3) = 1;
            x(1, 4) = 1;

            op.apply(x, Ax);

            THEN("The detector value is equal to the weight at the scaled distances")
            {
                data_t total_weight = 0;
                for (real_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{1.f, slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.bspline(detectorCoord / scaling);
                }

                CHECK_EQ(Ax[0], Approx(total_weight));
            }
        }
    }

    GIVEN("a single detector of size 1, at 45 degree")
    {
        std::vector<Geometry> geom;
        geom.emplace_back(stc, ctr, Degree{45}, VolumeData2D{Size2D{sizeDomain}},
                          SinogramData2D{Size2D{sizeRange}});

        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BSplineVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;

        WHEN("Setting only the voxels directly on the ray to 1")
        {
            // set all voxels on the ray to 1
            x(0, 0) = 1;
            x(1, 1) = 1;
            x(2, 2) = 1;
            x(3, 3) = 1;
            x(4, 4) = 1;

            op.apply(x, Ax);

            const auto weight = op.bspline(0);
            CAPTURE(weight);

            THEN("Each detector value is equal to 5 * the weight of distance 0")
            {
                CHECK_EQ(Ax[0], Approx(5 * weight));
            }
        }

        WHEN("Setting only the voxels directly above the ray to 1")
        {
            // set all voxels on the ray to 1
            x(0, 1) = 1;
            x(1, 2) = 1;
            x(2, 3) = 1;
            x(3, 4) = 1;

            op.apply(x, Ax);

            THEN("The detector value is equal to the weight at the scaled distances")
            {
                data_t total_weight = 0;
                for (real_t slice : {0, 1, 2, 3}) {
                    RealVector_t voxCoord{{slice, slice + 1}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.bspline(detectorCoord / scaling);
                }

                CHECK_EQ(Ax[0], Approx(total_weight));
            }
        }

        WHEN("Setting only the voxels directly below the ray to 1")
        {
            // set all voxels on the ray to 1
            x(1, 0) = 1;
            x(2, 1) = 1;
            x(3, 2) = 1;
            x(4, 3) = 1;

            op.apply(x, Ax);

            THEN("The detector value is equal to the weight at the scaled distances")
            {
                data_t total_weight = 0;
                for (real_t slice : {0, 1, 2, 3}) {
                    RealVector_t voxCoord{{slice + 1, slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.bspline(detectorCoord / scaling);
                }

                CHECK_EQ(Ax[0], Approx(total_weight));
            }
        }

        WHEN("Setting only the voxels two steps above the ray")
        {
            // set all voxels on the ray to 1
            x(0, 2) = 1;
            x(1, 3) = 1;
            x(2, 4) = 1;

            op.apply(x, Ax);

            THEN("The detector value is equal to the weight at the scaled distances")
            {
                data_t total_weight = 0;
                for (real_t slice : {0, 1, 2}) {
                    RealVector_t voxCoord{{slice, slice + 2}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.bspline(detectorCoord / scaling);
                }

                CHECK_EQ(Ax[0], Approx(total_weight));
            }
        }

        WHEN("Setting only the voxels two steps below the ray")
        {
            // set all voxels on the ray to 1
            x(2, 0) = 1;
            x(3, 1) = 1;
            x(4, 2) = 1;

            op.apply(x, Ax);

            THEN("The detector value is equal to the weight at the scaled distances")
            {
                data_t total_weight = 0;
                for (real_t slice : {0, 1, 2}) {
                    RealVector_t voxCoord{{slice + 2, slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.bspline(detectorCoord / scaling);
                }

                CHECK_EQ(Ax[0], Approx(total_weight));
            }
        }
    }

    GIVEN("a single detector of size 1, at 90 degree")
    {
        std::vector<Geometry> geom;
        geom.emplace_back(stc, ctr, Degree{90}, VolumeData2D{Size2D{sizeDomain}},
                          SinogramData2D{Size2D{sizeRange}});

        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BSplineVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;

        WHEN("Applying the BSplineVoxelProjectorCUDA<data_t> to the volume")
        {
            x = 0;
            // set all voxels on the ray to 1
            x(0, 2) = 1;
            x(1, 2) = 1;
            x(2, 2) = 1;
            x(3, 2) = 1;
            x(4, 2) = 1;

            op.apply(x, Ax);

            const auto weight = op.bspline(0);
            CAPTURE(weight);

            THEN("Each detector value is equal to 5 * the weight of distance 0")
            {
                CHECK_EQ(Ax[0], Approx(5 * weight));
            }
        }

        WHEN("Setting only the voxels direct neighbours above the ray to 1")
        {
            // set all voxels above the ray to 1, i.e the visited neighbours
            x(0, 3) = 1;
            x(1, 3) = 1;
            x(2, 3) = 1;
            x(3, 3) = 1;
            x(4, 3) = 1;

            op.apply(x, Ax);

            THEN("The detector value is equal to the weight at the scaled distances")
            {
                data_t total_weight = 0;
                for (real_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{slice, 3.f}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.bspline(detectorCoord / scaling);
                }

                CHECK_EQ(Ax[0], Approx(total_weight));
            }
        }

        WHEN("Setting only the voxels direct neighbours below the ray to 1")
        {
            // set all voxels above the ray to 1, i.e the visited neighbours
            x(0, 1) = 1;
            x(1, 1) = 1;
            x(2, 1) = 1;
            x(3, 1) = 1;
            x(4, 1) = 1;

            op.apply(x, Ax);

            THEN("The detector value is equal to the weight at the scaled distances")
            {
                data_t total_weight = 0;
                for (real_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{slice, 1.f}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.bspline(detectorCoord / scaling);
                }

                CHECK_EQ(Ax[0], Approx(total_weight));
            }
        }
    }

    GIVEN("a single detector of size 1, at 135 degree")
    {
        std::vector<Geometry> geom;
        geom.emplace_back(stc, ctr, Degree{135}, VolumeData2D{Size2D{sizeDomain}},
                          SinogramData2D{Size2D{sizeRange}});

        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BSplineVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;

        WHEN("Applying the BSplineVoxelProjectorCUDA<data_t> to the volume")
        {
            // set all voxels on the ray to 1
            x(0, 4) = 1;
            x(1, 3) = 1;
            x(2, 2) = 1;
            x(3, 1) = 1;
            x(4, 0) = 1;

            op.apply(x, Ax);

            const auto weight = op.bspline(0);
            CAPTURE(weight);

            THEN("Each detector value is equal to 5 * the weight of distance 0")
            {
                CHECK_EQ(Ax[0], Approx(5 * weight));
            }
        }

        WHEN("Setting only the voxels directly above on the ray to 1")
        {
            // set all voxels on the ray to 1
            x(1, 4) = 1;
            x(2, 3) = 1;
            x(3, 2) = 1;
            x(4, 1) = 1;

            op.apply(x, Ax);

            THEN("The detector value is equal to the weight at the scaled distances")
            {
                data_t total_weight = 0;
                for (real_t slice : {1, 2, 3, 4}) {
                    RealVector_t voxCoord{{slice, 5 - slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.bspline(detectorCoord / scaling);
                }

                CHECK_EQ(Ax[0], Approx(total_weight));
            }
        }

        WHEN("Setting only the voxels directly above on the ray to 1")
        {
            // set all voxels on the ray to 1
            x(0, 3) = 1;
            x(1, 2) = 1;
            x(2, 1) = 1;
            x(3, 0) = 1;

            op.apply(x, Ax);

            THEN("The detector value is equal to the weight at the scaled distances")
            {
                data_t total_weight = 0;
                for (real_t slice : {0, 1, 2, 3}) {
                    RealVector_t voxCoord{{slice, 3 - slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.bspline(detectorCoord / scaling);
                }

                CHECK_EQ(Ax[0], Approx(total_weight));
            }
        }

        WHEN("Setting only the voxels two above on the ray")
        {
            // set all voxels on the ray to 1
            x(2, 4) = 1;
            x(3, 3) = 1;
            x(4, 2) = 1;

            op.apply(x, Ax);

            THEN("The detector value is equal to the weight at the scaled distances")
            {
                data_t total_weight = 0;
                for (real_t slice : {2, 3, 4}) {
                    RealVector_t voxCoord{{slice, 6 - slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.bspline(detectorCoord / scaling);
                }

                CHECK_EQ(Ax[0], Approx(total_weight));
            }
        }

        WHEN("Setting only the voxels directly above on the ray to 1")
        {
            // set all voxels on the ray to 1
            x(0, 2) = 1;
            x(1, 1) = 1;
            x(2, 0) = 1;

            op.apply(x, Ax);

            THEN("The detector value is equal to the weight at the scaled distances")
            {
                data_t total_weight = 0;
                for (real_t slice : {0, 1, 2}) {
                    RealVector_t voxCoord{{slice, 2 - slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.bspline(detectorCoord / scaling);
                }

                CHECK_EQ(Ax[0], Approx(total_weight));
            }
        }
    }

    GIVEN("a single detector of size 1, at 180 degree")
    {
        std::vector<Geometry> geom;
        geom.emplace_back(stc, ctr, Degree{180}, VolumeData2D{Size2D{sizeDomain}},
                          SinogramData2D{Size2D{sizeRange}});

        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BSplineVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;

        WHEN("Applying the BSplineVoxelProjectorCUDA to the volume")
        {
            x = 0;
            // set all voxels on the ray to 1
            x(2, 0) = 1;
            x(2, 1) = 1;
            x(2, 2) = 1;
            x(2, 3) = 1;
            x(2, 4) = 1;

            op.apply(x, Ax);

            const auto weight = op.bspline(0);
            CAPTURE(weight);

            THEN("Each detector value is equal to 5 * the weight of distance 0")
            {
                CHECK_EQ(Ax[0], Approx(5 * weight));
            }
        }

        WHEN("Setting only the voxels direct neighbours above the ray to 1")
        {
            // set all voxels above the ray to 1, i.e the visited neighbours
            x(3, 0) = 1;
            x(3, 1) = 1;
            x(3, 2) = 1;
            x(3, 3) = 1;
            x(3, 4) = 1;

            op.apply(x, Ax);

            THEN("The detector value is equal to the weight at the scaled distances")
            {
                data_t total_weight = 0;
                for (real_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{3.f, slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.bspline(detectorCoord / scaling);
                }

                CHECK_EQ(Ax[0], Approx(total_weight));
            }
        }

        WHEN("Setting only the voxels direct neighbours below the ray to 1")
        {
            // set all voxels above the ray to 1, i.e the visited neighbours
            x(1, 0) = 1;
            x(1, 1) = 1;
            x(1, 2) = 1;
            x(1, 3) = 1;
            x(1, 4) = 1;

            op.apply(x, Ax);

            THEN("The detector value is equal to the weight at the scaled distances")
            {
                data_t total_weight = 0;
                for (real_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{1.f, slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.bspline(detectorCoord / scaling);
                }

                CHECK_EQ(Ax[0], Approx(total_weight));
            }
        }
    }

    GIVEN("a single detector of size 1, at 225 degree")
    {
        std::vector<Geometry> geom;
        geom.emplace_back(stc, ctr, Degree{225}, VolumeData2D{Size2D{sizeDomain}},
                          SinogramData2D{Size2D{sizeRange}});

        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BSplineVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;

        WHEN("Applying the BSplineVoxelProjectorCUDA to the volume")
        {
            // set all voxels on the ray to 1
            x(0, 0) = 1;
            x(1, 1) = 1;
            x(2, 2) = 1;
            x(3, 3) = 1;
            x(4, 4) = 1;

            op.apply(x, Ax);

            const auto weight = op.bspline(0);
            CAPTURE(weight);

            THEN("Each detector value is equal to 5 * the weight of distance 0")
            {
                CHECK_EQ(Ax[0], Approx(5 * weight));
            }
        }
    }

    GIVEN("a single detector of size 1, at 270 degree")
    {
        std::vector<Geometry> geom;
        geom.emplace_back(stc, ctr, Degree{270}, VolumeData2D{Size2D{sizeDomain}},
                          SinogramData2D{Size2D{sizeRange}});

        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BSplineVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;

        WHEN("Applying the BSplineVoxelProjectorCUDA to the volume")
        {
            x = 0;
            // set all voxels on the ray to 1
            x(0, 2) = 1;
            x(1, 2) = 1;
            x(2, 2) = 1;
            x(3, 2) = 1;
            x(4, 2) = 1;

            op.apply(x, Ax);

            const auto weight = op.bspline(0);
            CAPTURE(weight);

            THEN("Each detector value is equal to 5 * the weight of distance 0")
            {
                CHECK_EQ(Ax[0], Approx(5 * weight));
            }
        }
    }

    GIVEN("a single detector of size 1, at 315 degree")
    {
        std::vector<Geometry> geom;
        geom.emplace_back(stc, ctr, Degree{315}, VolumeData2D{Size2D{sizeDomain}},
                          SinogramData2D{Size2D{sizeRange}});

        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BSplineVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;

        WHEN("Applying the BSplineVoxelProjectorCUDA to the volume")
        {
            // set all voxels on the ray to 1
            x(0, 4) = 1;
            x(1, 3) = 1;
            x(2, 2) = 1;
            x(3, 1) = 1;
            x(4, 0) = 1;

            op.apply(x, Ax);

            const auto weight = op.bspline(0);
            CAPTURE(weight);

            THEN("Each detector value is equal to 5 * the weight of distance 0")
            {
                CHECK_EQ(Ax[0], Approx(5 * weight));
            }
        }
    }
}

TEST_CASE_TEMPLATE("BSplineVoxelProjectorCUDA: Test backward projection", data_t, float, double)
{
    const IndexVector_t sizeDomain({{5, 5}});
    const IndexVector_t sizeRange({{1, 1}});

    auto domain = VolumeDescriptor(sizeDomain);
    auto x = DataContainer<data_t>(domain);

    auto stc = SourceToCenterOfRotation{100};
    auto ctr = CenterOfRotationToDetector{5};
    auto volData = VolumeData2D{Size2D{sizeDomain}};
    auto sinoData = SinogramData2D{Size2D{sizeRange}};

    GIVEN("a single detector of size 1, at 0 degree")
    {
        std::vector<Geometry> geom;
        geom.emplace_back(stc, ctr, Degree{0}, VolumeData2D{Size2D{sizeDomain}},
                          SinogramData2D{Size2D{sizeRange}});

        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BSplineVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;
        x = 0;

        WHEN("Backward projecting the operator")
        {
            Ax[0] = 1;

            op.applyAdjoint(Ax, x);

            THEN("The main direction of the ray is equal to the weight")
            {
                const auto weight = op.bspline(0);
                CAPTURE(weight);

                CHECK_EQ(x(2, 0), Approx(weight));
                CHECK_EQ(x(2, 1), Approx(weight));
                CHECK_EQ(x(2, 2), Approx(weight));
                CHECK_EQ(x(2, 3), Approx(weight));
                CHECK_EQ(x(2, 4), Approx(weight));
            }

            THEN("The pixel 1 above the main direction of the ray have the projected weight")
            {
                for (index_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{3.f, static_cast<float>(slice)}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.bspline(detectorCoord / scaling);
                    CHECK_EQ(x(3, slice), Approx(weight));
                }
            }
            THEN("The pixel 1 below the main direction of the ray have the projected weight")
            {
                for (index_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{1.f, static_cast<float>(slice)}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.bspline(detectorCoord / scaling);
                    CHECK_EQ(x(1, slice), Approx(weight));
                }
            }
        }
    }

    GIVEN("a single detector of size 1, at 180 degree")
    {
        std::vector<Geometry> geom;
        geom.emplace_back(stc, ctr, Degree{180}, VolumeData2D{Size2D{sizeDomain}},
                          SinogramData2D{Size2D{sizeRange}});

        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BSplineVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;
        x = 0;

        WHEN("Backward projecting the operator")
        {
            Ax[0] = 1;

            op.applyAdjoint(Ax, x);

            THEN("The main direction of the ray is equal to the weight")
            {
                const auto weight = op.bspline(0);
                CAPTURE(weight);

                CHECK_EQ(x(2, 0), Approx(weight));
                CHECK_EQ(x(2, 1), Approx(weight));
                CHECK_EQ(x(2, 2), Approx(weight));
                CHECK_EQ(x(2, 3), Approx(weight));
                CHECK_EQ(x(2, 4), Approx(weight));
            }

            THEN("The pixel 1 above the main direction of the ray have the projected weight")
            {
                for (index_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{3.f, static_cast<float>(slice)}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.bspline(detectorCoord / scaling);
                    CHECK_EQ(x(3, slice), Approx(weight));
                }
            }
            THEN("The pixel 1 below the main direction of the ray have the projected weight")
            {
                for (index_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{1.f, static_cast<float>(slice)}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.bspline(detectorCoord / scaling);
                    CHECK_EQ(x(1, slice), Approx(weight));
                }
            }
        }
    }

    GIVEN("a single detector of size 1, at 90 degree")
    {
        std::vector<Geometry> geom;
        geom.emplace_back(stc, ctr, Degree{90}, VolumeData2D{Size2D{sizeDomain}},
                          SinogramData2D{Size2D{sizeRange}});

        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BSplineVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;
        x = 0;

        WHEN("Backward projecting the operator")
        {
            Ax[0] = 1;

            op.applyAdjoint(Ax, x);

            THEN("The main direction of the ray is equal to the weight")
            {
                const auto weight = op.bspline(0);
                CAPTURE(weight);

                CHECK_EQ(x(0, 2), Approx(weight));
                CHECK_EQ(x(1, 2), Approx(weight));
                CHECK_EQ(x(2, 2), Approx(weight));
                CHECK_EQ(x(3, 2), Approx(weight));
                CHECK_EQ(x(4, 2), Approx(weight));
            }

            THEN("The pixel 1 above the main direction of the ray have the projected weight")
            {
                for (index_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{static_cast<float>(slice), 3.f}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.bspline(detectorCoord / scaling);
                    CHECK_EQ(x(slice, 3), Approx(weight));
                }
            }
            THEN("The pixel 1 below the main direction of the ray have the projected weight")
            {
                for (index_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{static_cast<float>(slice), 1.f}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.bspline(detectorCoord / scaling);
                    CHECK_EQ(x(slice, 1), Approx(weight));
                }
            }
        }
    }

    GIVEN("a single detector of size 1, at 270 degree")
    {
        std::vector<Geometry> geom;
        geom.emplace_back(stc, ctr, Degree{270}, VolumeData2D{Size2D{sizeDomain}},
                          SinogramData2D{Size2D{sizeRange}});

        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BSplineVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;
        x = 0;

        WHEN("Backward projecting the operator")
        {
            Ax[0] = 1;

            op.applyAdjoint(Ax, x);

            THEN("The main direction of the ray is equal to the weight")
            {
                const auto weight = op.bspline(0);
                CAPTURE(weight);

                CHECK_EQ(x(0, 2), Approx(weight));
                CHECK_EQ(x(1, 2), Approx(weight));
                CHECK_EQ(x(2, 2), Approx(weight));
                CHECK_EQ(x(3, 2), Approx(weight));
                CHECK_EQ(x(4, 2), Approx(weight));
            }
            THEN("The pixel 1 above the main direction of the ray have the projected weight")
            {
                for (index_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{static_cast<float>(slice), 3.f}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.bspline(detectorCoord / scaling);
                    CHECK_EQ(x(slice, 3), Approx(weight));
                }
            }
            THEN("The pixel 1 below the main direction of the ray have the projected weight")
            {
                for (index_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{static_cast<float>(slice), 1.f}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.bspline(detectorCoord / scaling);
                    CHECK_EQ(x(slice, 1), Approx(weight));
                }
            }
        }
    }

    GIVEN("a single detector of size 1, at 45 degree")
    {
        std::vector<Geometry> geom;
        geom.emplace_back(stc, ctr, Degree{45}, VolumeData2D{Size2D{sizeDomain}},
                          SinogramData2D{Size2D{sizeRange}});

        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BSplineVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;
        x = 0;

        WHEN("Backward projecting the operator")
        {
            Ax[0] = 1;

            op.applyAdjoint(Ax, x);

            THEN("The main direction of the ray is equal to the weight")
            {
                const auto weight = op.bspline(0);
                CAPTURE(weight);

                CHECK_EQ(x(0, 0), Approx(weight));
                CHECK_EQ(x(1, 1), Approx(weight));
                CHECK_EQ(x(2, 2), Approx(weight));
                CHECK_EQ(x(3, 3), Approx(weight));
                CHECK_EQ(x(4, 4), Approx(weight));
            }

            THEN("The pixel 1 above the main direction of the ray have the projected weight")
            {
                for (index_t slice : {0, 1, 2, 3}) {
                    auto sliceF = static_cast<float>(slice);
                    RealVector_t voxCoord{{sliceF, sliceF + 1}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.bspline(detectorCoord / scaling);
                    CHECK_EQ(x(slice, slice + 1), Approx(weight));
                }
            }

            THEN("The pixel 1 below the main direction of the ray have the projected weight")
            {
                for (index_t slice : {0, 1, 2, 3}) {
                    auto sliceF = static_cast<float>(slice);
                    RealVector_t voxCoord{{sliceF + 1, sliceF}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.bspline(detectorCoord / scaling);
                    CHECK_EQ(x(slice + 1, slice), Approx(weight));
                }
            }

            THEN("The pixel 2 above the main direction of the ray have the projected weight")
            {
                for (index_t slice : {0, 1, 2}) {
                    auto sliceF = static_cast<float>(slice);
                    RealVector_t voxCoord{{sliceF, 2 - sliceF}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.bspline(detectorCoord / scaling);
                    CHECK_EQ(x(slice, 2 - slice), Approx(weight));
                }
            }

            THEN("The pixel 2 above the main direction of the ray have the projected weight")
            {
                for (index_t slice : {0, 1, 2}) {
                    auto sliceF = static_cast<float>(slice);
                    RealVector_t voxCoord{{2 - sliceF, sliceF}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.bspline(detectorCoord / scaling);
                    CHECK_EQ(x(2 - slice, slice), Approx(weight));
                }
            }
        }
    }
}

TEST_SUITE_END();