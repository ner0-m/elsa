#include "doctest/doctest.h"

#include "VoxelProjector.h"
#include "PlanarDetectorDescriptor.h"

#include "PrettyPrint/Eigen.h"
#include "PrettyPrint/Stl.h"
#include <spdlog/fmt/fmt.h>

using namespace elsa;
using namespace elsa::geometry;
using namespace doctest;

TEST_SUITE_BEGIN("projectors");

Eigen::IOFormat vecfmt(10, 0, ", ", ", ", "", "", "[", "]");
Eigen::IOFormat matfmt(10, 0, ", ", "\n", "\t\t[", "]");

TYPE_TO_STRING(BlobVoxelProjector<float>);

// Redefine GIVEN such that it's nicely usable inside an loop
#undef GIVEN
#define GIVEN(...) DOCTEST_SUBCASE((std::string("   Given: ") + std::string(__VA_ARGS__)).c_str())

TEST_CASE_TEMPLATE("BlobVoxelProjector: Testing simple volume 3D with one detector pixel", data_t,
                   float, double)
{
    const IndexVector_t sizeDomain({{5, 5, 5}});
    const IndexVector_t sizeRange({{1, 1, 1}});

    auto stc = SourceToCenterOfRotation{100};
    auto ctr = CenterOfRotationToDetector{5};

    for (int i = 1; i < 4; i++) {
        real_t scaling = i / 2.0f;
        const RealVector_t spacing{{scaling, scaling, scaling}};
        GIVEN("Spacing of " + std::to_string(scaling))
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
                auto op = BlobVoxelProjector<data_t>(domain, range);

                auto Ax = DataContainer<data_t>(range);
                Ax = 0;

                WHEN("projecting forward and only the center voxel is set to 1")
                {
                    op.apply(x, Ax);

                    const auto weight = op.weight(0);
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
}

TEST_CASE_TEMPLATE("BlobVoxelProjector: Testing simple volume 2D with one detector Pixel", data_t,
                   float, double)
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
                auto op = BlobVoxelProjector<data_t>(domain, range);

                auto Ax = DataContainer<data_t>(range);
                Ax = 0;

                WHEN("projecting forward and only the center voxel is set to 1")
                {
                    // set center voxel to 1
                    x(2, 2) = 1;

                    op.apply(x, Ax);

                    const auto weight = op.weight(0);
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

TEST_CASE_TEMPLATE("BlobVoxelProjector: Testing simple volume 2D with two detector pixels", data_t,
                   float, double)
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
            auto op = BlobVoxelProjector<data_t>(domain, range);

            auto Ax = DataContainer<data_t>(range);
            Ax = 0;

            WHEN("only the center voxel is set to 1")
            {
                // set center voxel to 1
                x(2, 2) = 1;

                op.apply(x, Ax);

                const auto weight_dist_half = op.weight(0.4761878);
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

TEST_CASE_TEMPLATE("BlobVoxelProjector: Testing simple volume 2D with three detector pixels",
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
            auto op = BlobVoxelProjector<data_t>(domain, range);

            auto Ax = DataContainer<data_t>(range);
            Ax = 0;

            WHEN("only the center voxel is set to 1")
            {
                // set center voxel to 1
                x(2, 2) = 1;

                op.apply(x, Ax);

                const auto weight_dist0 = op.weight(0);
                const auto weight_dist1 = op.weight(0.95233786);
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

TEST_CASE("BlobVoxelProjector: Test single detector pixel")
{
    const IndexVector_t sizeDomain({{5, 5}});
    const IndexVector_t sizeRange({{1, 1}});

    auto domain = VolumeDescriptor(sizeDomain);
    auto x = DataContainer(domain);
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
        auto op = BlobVoxelProjector(domain, range);

        auto Ax = DataContainer(range);
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

            const auto weight = op.weight(0);
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
                real_t total_weight = 0;
                for (real_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{3.f, slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.weight(detectorCoord / scaling);
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
                real_t total_weight = 0;
                for (real_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{1.f, slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.weight(detectorCoord / scaling);
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
        auto op = BlobVoxelProjector(domain, range);

        auto Ax = DataContainer(range);
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

            const auto weight = op.weight(0);
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
                real_t total_weight = 0;
                for (real_t slice : {0, 1, 2, 3}) {
                    RealVector_t voxCoord{{slice, slice + 1}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.weight(detectorCoord / scaling);
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
                real_t total_weight = 0;
                for (real_t slice : {0, 1, 2, 3}) {
                    RealVector_t voxCoord{{slice + 1, slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.weight(detectorCoord / scaling);
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
                real_t total_weight = 0;
                for (real_t slice : {0, 1, 2}) {
                    RealVector_t voxCoord{{slice, slice + 2}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.weight(detectorCoord / scaling);
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
                real_t total_weight = 0;
                for (real_t slice : {0, 1, 2}) {
                    RealVector_t voxCoord{{slice + 2, slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.weight(detectorCoord / scaling);
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
        auto op = BlobVoxelProjector(domain, range);

        auto Ax = DataContainer(range);
        Ax = 0;

        WHEN("Applying the BlobVoxelProjector to the volume")
        {
            x = 0;
            // set all voxels on the ray to 1
            x(0, 2) = 1;
            x(1, 2) = 1;
            x(2, 2) = 1;
            x(3, 2) = 1;
            x(4, 2) = 1;

            op.apply(x, Ax);

            const auto weight = op.weight(0);
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
                real_t total_weight = 0;
                for (real_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{slice, 3.f}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.weight(detectorCoord / scaling);
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
                real_t total_weight = 0;
                for (real_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{slice, 1.f}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.weight(detectorCoord / scaling);
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
        auto op = BlobVoxelProjector(domain, range);

        auto Ax = DataContainer(range);
        Ax = 0;

        WHEN("Applying the BlobVoxelProjector to the volume")
        {
            // set all voxels on the ray to 1
            x(0, 4) = 1;
            x(1, 3) = 1;
            x(2, 2) = 1;
            x(3, 1) = 1;
            x(4, 0) = 1;

            op.apply(x, Ax);

            const auto weight = op.weight(0);
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
                real_t total_weight = 0;
                for (real_t slice : {1, 2, 3, 4}) {
                    RealVector_t voxCoord{{slice, 5 - slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.weight(detectorCoord / scaling);
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
                real_t total_weight = 0;
                for (real_t slice : {0, 1, 2, 3}) {
                    RealVector_t voxCoord{{slice, 3 - slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.weight(detectorCoord / scaling);
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
                real_t total_weight = 0;
                for (real_t slice : {2, 3, 4}) {
                    RealVector_t voxCoord{{slice, 6 - slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.weight(detectorCoord / scaling);
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
                real_t total_weight = 0;
                for (real_t slice : {0, 1, 2}) {
                    RealVector_t voxCoord{{slice, 2 - slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.weight(detectorCoord / scaling);
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
        auto op = BlobVoxelProjector(domain, range);

        auto Ax = DataContainer(range);
        Ax = 0;

        WHEN("Applying the BlobVoxelProjector to the volume")
        {
            x = 0;
            // set all voxels on the ray to 1
            x(2, 0) = 1;
            x(2, 1) = 1;
            x(2, 2) = 1;
            x(2, 3) = 1;
            x(2, 4) = 1;

            op.apply(x, Ax);

            const auto weight = op.weight(0);
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
                real_t total_weight = 0;
                for (real_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{3.f, slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.weight(detectorCoord / scaling);
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
                real_t total_weight = 0;
                for (real_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{1.f, slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.weight(detectorCoord / scaling);
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
        auto op = BlobVoxelProjector(domain, range);

        auto Ax = DataContainer(range);
        Ax = 0;

        WHEN("Applying the BlobVoxelProjector to the volume")
        {
            // set all voxels on the ray to 1
            x(0, 0) = 1;
            x(1, 1) = 1;
            x(2, 2) = 1;
            x(3, 3) = 1;
            x(4, 4) = 1;

            op.apply(x, Ax);

            const auto weight = op.weight(0);
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
        auto op = BlobVoxelProjector(domain, range);

        auto Ax = DataContainer(range);
        Ax = 0;

        WHEN("Applying the BlobVoxelProjector to the volume")
        {
            x = 0;
            // set all voxels on the ray to 1
            x(0, 2) = 1;
            x(1, 2) = 1;
            x(2, 2) = 1;
            x(3, 2) = 1;
            x(4, 2) = 1;

            op.apply(x, Ax);

            const auto weight = op.weight(0);
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
        auto op = BlobVoxelProjector(domain, range);

        auto Ax = DataContainer(range);
        Ax = 0;

        WHEN("Applying the BlobVoxelProjector to the volume")
        {
            // set all voxels on the ray to 1
            x(0, 4) = 1;
            x(1, 3) = 1;
            x(2, 2) = 1;
            x(3, 1) = 1;
            x(4, 0) = 1;

            op.apply(x, Ax);

            const auto weight = op.weight(0);
            CAPTURE(weight);

            THEN("Each detector value is equal to 5 * the weight of distance 0")
            {
                CHECK_EQ(Ax[0], Approx(5 * weight));
            }
        }
    }
}

TEST_CASE("BlobVoxelProjector: Test single detector pixel")
{
    const IndexVector_t sizeDomain({{5, 5}});
    const IndexVector_t sizeRange({{1, 1}});

    auto domain = VolumeDescriptor(sizeDomain);
    auto x = DataContainer(domain);
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
        auto op = BlobVoxelProjector(domain, range);

        auto Ax = DataContainer(range);
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

            const auto weight = op.weight(0);
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
                real_t total_weight = 0;
                for (real_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{3.f, slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.weight(detectorCoord / scaling);
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
                real_t total_weight = 0;
                for (real_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{1.f, slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.weight(detectorCoord / scaling);
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
        auto op = BlobVoxelProjector(domain, range);

        auto Ax = DataContainer(range);
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

            const auto weight = op.weight(0);
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
                real_t total_weight = 0;
                for (real_t slice : {0, 1, 2, 3}) {
                    RealVector_t voxCoord{{slice, slice + 1}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.weight(detectorCoord / scaling);
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
                real_t total_weight = 0;
                for (real_t slice : {0, 1, 2, 3}) {
                    RealVector_t voxCoord{{slice + 1, slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.weight(detectorCoord / scaling);
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
                real_t total_weight = 0;
                for (real_t slice : {0, 1, 2}) {
                    RealVector_t voxCoord{{slice, slice + 2}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.weight(detectorCoord / scaling);
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
                real_t total_weight = 0;
                for (real_t slice : {0, 1, 2}) {
                    RealVector_t voxCoord{{slice + 2, slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.weight(detectorCoord / scaling);
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
        auto op = BlobVoxelProjector(domain, range);

        auto Ax = DataContainer(range);
        Ax = 0;

        WHEN("Applying the BlobVoxelProjector to the volume")
        {
            x = 0;
            // set all voxels on the ray to 1
            x(0, 2) = 1;
            x(1, 2) = 1;
            x(2, 2) = 1;
            x(3, 2) = 1;
            x(4, 2) = 1;

            op.apply(x, Ax);

            const auto weight = op.weight(0);
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
                real_t total_weight = 0;
                for (real_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{slice, 3.f}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.weight(detectorCoord / scaling);
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
                real_t total_weight = 0;
                for (real_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{slice, 1.f}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.weight(detectorCoord / scaling);
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
        auto op = BlobVoxelProjector(domain, range);

        auto Ax = DataContainer(range);
        Ax = 0;

        WHEN("Applying the BlobVoxelProjector to the volume")
        {
            // set all voxels on the ray to 1
            x(0, 4) = 1;
            x(1, 3) = 1;
            x(2, 2) = 1;
            x(3, 1) = 1;
            x(4, 0) = 1;

            op.apply(x, Ax);

            const auto weight = op.weight(0);
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
                real_t total_weight = 0;
                for (real_t slice : {1, 2, 3, 4}) {
                    RealVector_t voxCoord{{slice, 5 - slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.weight(detectorCoord / scaling);
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
                real_t total_weight = 0;
                for (real_t slice : {0, 1, 2, 3}) {
                    RealVector_t voxCoord{{slice, 3 - slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.weight(detectorCoord / scaling);
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
                real_t total_weight = 0;
                for (real_t slice : {2, 3, 4}) {
                    RealVector_t voxCoord{{slice, 6 - slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.weight(detectorCoord / scaling);
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
                real_t total_weight = 0;
                for (real_t slice : {0, 1, 2}) {
                    RealVector_t voxCoord{{slice, 2 - slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.weight(detectorCoord / scaling);
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
        auto op = BlobVoxelProjector(domain, range);

        auto Ax = DataContainer(range);
        Ax = 0;

        WHEN("Applying the BlobVoxelProjector to the volume")
        {
            x = 0;
            // set all voxels on the ray to 1
            x(2, 0) = 1;
            x(2, 1) = 1;
            x(2, 2) = 1;
            x(2, 3) = 1;
            x(2, 4) = 1;

            op.apply(x, Ax);

            const auto weight = op.weight(0);
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
                real_t total_weight = 0;
                for (real_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{3.f, slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.weight(detectorCoord / scaling);
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
                real_t total_weight = 0;
                for (real_t slice : {0, 1, 2, 3, 4}) {
                    RealVector_t voxCoord{{1.f, slice}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    total_weight += op.weight(detectorCoord / scaling);
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
        auto op = BlobVoxelProjector(domain, range);

        auto Ax = DataContainer(range);
        Ax = 0;

        WHEN("Applying the BlobVoxelProjector to the volume")
        {
            // set all voxels on the ray to 1
            x(0, 0) = 1;
            x(1, 1) = 1;
            x(2, 2) = 1;
            x(3, 3) = 1;
            x(4, 4) = 1;

            op.apply(x, Ax);

            const auto weight = op.weight(0);
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
        auto op = BlobVoxelProjector(domain, range);

        auto Ax = DataContainer(range);
        Ax = 0;

        WHEN("Applying the BlobVoxelProjector to the volume")
        {
            x = 0;
            // set all voxels on the ray to 1
            x(0, 2) = 1;
            x(1, 2) = 1;
            x(2, 2) = 1;
            x(3, 2) = 1;
            x(4, 2) = 1;

            op.apply(x, Ax);

            const auto weight = op.weight(0);
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
        auto op = BlobVoxelProjector(domain, range);

        auto Ax = DataContainer(range);
        Ax = 0;

        WHEN("Applying the BlobVoxelProjector to the volume")
        {
            // set all voxels on the ray to 1
            x(0, 4) = 1;
            x(1, 3) = 1;
            x(2, 2) = 1;
            x(3, 1) = 1;
            x(4, 0) = 1;

            op.apply(x, Ax);

            const auto weight = op.weight(0);
            CAPTURE(weight);

            THEN("Each detector value is equal to 5 * the weight of distance 0")
            {
                CHECK_EQ(Ax[0], Approx(5 * weight));
            }
        }
    }
}

TEST_CASE("BlobVoxelProjector: Test backward projection")
{
    const IndexVector_t sizeDomain({{5, 5}});
    const IndexVector_t sizeRange({{1, 1}});

    auto domain = VolumeDescriptor(sizeDomain);
    auto x = DataContainer(domain);

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
        auto op = BlobVoxelProjector(domain, range);

        auto Ax = DataContainer(range);
        Ax = 0;
        x = 0;

        WHEN("Backward projecting the operator")
        {
            Ax[0] = 1;

            op.applyAdjoint(Ax, x);

            THEN("The main direction of the ray is equal to the weight")
            {
                const auto weight = op.weight(0);
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
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.weight(detectorCoord / scaling);
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
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.weight(detectorCoord / scaling);
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
        auto op = BlobVoxelProjector(domain, range);

        auto Ax = DataContainer(range);
        Ax = 0;
        x = 0;

        WHEN("Backward projecting the operator")
        {
            Ax[0] = 1;

            op.applyAdjoint(Ax, x);

            THEN("The main direction of the ray is equal to the weight")
            {
                const auto weight = op.weight(0);
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
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.weight(detectorCoord / scaling);
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
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.weight(detectorCoord / scaling);
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
        auto op = BlobVoxelProjector(domain, range);

        auto Ax = DataContainer(range);
        Ax = 0;
        x = 0;

        WHEN("Backward projecting the operator")
        {
            Ax[0] = 1;

            op.applyAdjoint(Ax, x);

            THEN("The main direction of the ray is equal to the weight")
            {
                const auto weight = op.weight(0);
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
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.weight(detectorCoord / scaling);
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
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.weight(detectorCoord / scaling);
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
        auto op = BlobVoxelProjector(domain, range);

        auto Ax = DataContainer(range);
        Ax = 0;
        x = 0;

        WHEN("Backward projecting the operator")
        {
            Ax[0] = 1;

            op.applyAdjoint(Ax, x);

            THEN("The main direction of the ray is equal to the weight")
            {
                const auto weight = op.weight(0);
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
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.weight(detectorCoord / scaling);
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
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.weight(detectorCoord / scaling);
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
        auto op = BlobVoxelProjector(domain, range);

        auto Ax = DataContainer(range);
        Ax = 0;
        x = 0;

        WHEN("Backward projecting the operator")
        {
            Ax[0] = 1;

            op.applyAdjoint(Ax, x);

            THEN("The main direction of the ray is equal to the weight")
            {
                const auto weight = op.weight(0);
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
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.weight(detectorCoord / scaling);
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
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.weight(detectorCoord / scaling);
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
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.weight(detectorCoord / scaling);
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
                    real_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.weight(detectorCoord / scaling);
                    CHECK_EQ(x(2 - slice, slice), Approx(weight));
                }
            }
        }
    }
}

TEST_CASE_TEMPLATE("BlobVoxelProjector: Test weights", data_t, float, double)
{

    std::array<double, 51> expected{
        1.3671064952680276,     1.3635202864368146,    1.3528128836429958,
        1.3351368521026497,     1.3107428112196733,    1.2799740558068384,
        1.2432592326454344,     1.2011032712842662,    1.1540768137691881,
        1.1028044260488254,     1.0479519029788988,    0.9902129983165086,
        0.930295920364307,      0.868909932853028,     0.80675238945173,
        0.7444965095220905,     0.6827801732549599,    0.6221959772930218,
        0.5632827487344612,     0.5065186675909519,    0.4523160970195944,
        0.40101816870068707,    0.35289711932154216,   0.30815432491147815,
        0.2669219342875695,     0.22926596246745282,   0.1951906707135796,
        0.1646440327638897,     0.13752406736650655,   0.11368580576665363,
        0.09294865929450916,    0.0751039563916854,    0.05992242974801804,
        0.04716145192475515,    0.036571840947646775,  0.027904084748497336,
        0.020913863801848218,   0.015366783581446854,  0.01104226128798172,
        0.007736543464620423,   0.005264861504239613,  0.003462759678911652,
        0.0021866543689359847,  0.0013137030013652602, 0.0007410763873099119,
        0.0003847384204545552,  0.0001778423521946198, 6.885294293780879e-05,
        1.9497773431607567e-05, 2.632583006403572e-06, 0};

    const IndexVector_t sizeDomain({{5, 5}});
    const IndexVector_t sizeRange({{1, 1}});

    auto domain = VolumeDescriptor(sizeDomain);
    auto x = DataContainer(domain);
    x = 1;

    auto stc = SourceToCenterOfRotation{100};
    auto ctr = CenterOfRotationToDetector{5};
    auto volData = VolumeData2D{Size2D{sizeDomain}};
    auto sinoData = SinogramData2D{Size2D{sizeRange}};

    std::vector<Geometry> geom;
    geom.emplace_back(stc, ctr, Degree{0}, std::move(volData), std::move(sinoData));

    auto range = PlanarDetectorDescriptor(sizeRange, geom);
    auto op = BlobVoxelProjector(domain, range);

    for (int i = 0; i < 50; ++i) {
        const auto distance = i / 25.;

        CAPTURE(i);
        CAPTURE(distance);
        CHECK_EQ(Approx(op.weight(distance)), expected[i]);
    }
}

TEST_SUITE_END();
