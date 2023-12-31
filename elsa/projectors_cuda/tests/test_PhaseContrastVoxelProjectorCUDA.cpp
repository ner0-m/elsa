#include "doctest/doctest.h"

#include "VoxelProjectorCUDA.h"
#include "PlanarDetectorDescriptor.h"

#include "PrettyPrint/Eigen.h"
#include "PrettyPrint/Stl.h"
#include "spdlog/fmt/bundled/core.h"

using namespace elsa;
using namespace elsa::geometry;
using namespace doctest;

TEST_SUITE_BEGIN("phase contrast projector cuda");

Eigen::IOFormat vecfmt(10, 0, ", ", ", ", "", "", "[", "]");
Eigen::IOFormat matfmt(10, 0, ", ", "\n", "\t\t[", "]");

TYPE_TO_STRING(PhaseContrastBlobVoxelProjectorCUDA<float>);

// Redefine GIVEN such that it's nicely usable inside an loop
#undef GIVEN
#define GIVEN(...) DOCTEST_SUBCASE((std::string("   Given: ") + std::string(__VA_ARGS__)).c_str())

double distance_from_center(int ray_number)
{
    return std::sqrt(ray_number * ray_number / 2.0) * math::sgn(ray_number);
}

double distance_from_center(IndexVector_t p, RealVector_t middle, float alpha)
{
    RealVector_t rd{{std::cos(alpha), std::sin(alpha)}};
    RealRay_t ray{p.template cast<real_t>(), rd};
    RealVector_t rMiddle = middle - ray.projection(middle);
    // embed vectors in 3D and get the z component of a cross product
    auto dir = rd[0] * rMiddle[1] - rd[1] * rMiddle[0];
    return dir;
}

TEST_CASE_TEMPLATE(
    "PhaseContrastBlobVoxelProjectorCUDA: Testing simple volume 2D with two detector pixels",
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
            auto op = PhaseContrastBlobVoxelProjectorCUDA<data_t>(domain, range);

            auto Ax = DataContainer<data_t>(range);
            Ax = 0;

            WHEN("only the center voxel is set to 1")
            {
                // set center voxel to 1
                x(2, 2) = 1;

                op.apply(x, Ax);

                const auto weight_dist_half = op.blob.derivative(0.4761878);
                CAPTURE(weight_dist_half);

                THEN("Detector values are reverse")
                {
                    CAPTURE(Ax[0]);
                    CAPTURE(Ax[1]);
                    CHECK_EQ(weight_dist_half, Approx(Ax[0]).epsilon(0.01));
                    CHECK_EQ(-weight_dist_half, Approx(Ax[1]).epsilon(0.01));
                }
            }
        }
    }
}

TEST_CASE_TEMPLATE(
    "PhaseContrastBlobVoxelProjectorCUDA: Testing simple volume 2D with three detector pixels",
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
            auto op = PhaseContrastBlobVoxelProjectorCUDA<data_t>(domain, range);

            auto Ax = DataContainer<data_t>(range);
            Ax = 0;

            WHEN("only the center voxel is set to 1")
            {
                // set center voxel to 1
                x(2, 2) = 1;

                op.apply(x, Ax);

                const auto weight_dist0 = op.blob.derivative(0);
                const auto weight_dist1 = op.blob.derivative(0.95233786);
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
                    CHECK_EQ(-weight_dist1, Approx(Ax[2]).epsilon(0.01));
                }
            }
        }
    }
}

TEST_CASE_TEMPLATE("PhaseContrastBlobVoxelProjectorCUDA: Test single detector pixel", data_t, float,
                   double)
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
        auto op = PhaseContrastBlobVoxelProjectorCUDA<data_t>(domain, range);

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

            const auto weight = op.blob.derivative(0);
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
                    total_weight += op.blob.derivative(detectorCoord / scaling);
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
                    total_weight += op.blob.derivative(detectorCoord / scaling);
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
        auto op = PhaseContrastBlobVoxelProjectorCUDA<data_t>(domain, range);

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

            const auto weight = op.blob.derivative(0);
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
                    total_weight += op.blob.derivative(detectorCoord / scaling);
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
                    total_weight += op.blob.derivative(detectorCoord / scaling);
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
                    total_weight += op.blob.derivative(detectorCoord / scaling);
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
                    total_weight += op.blob.derivative(detectorCoord / scaling);
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
        auto op = PhaseContrastBlobVoxelProjectorCUDA<data_t>(domain, range);

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

            const auto weight = op.blob.derivative(0);
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
                    total_weight += op.blob.derivative(detectorCoord / scaling);
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
                    total_weight += op.blob.derivative(detectorCoord / scaling);
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
        auto op = PhaseContrastBlobVoxelProjectorCUDA<data_t>(domain, range);

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

            const auto weight = op.blob.derivative(0);
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
                    total_weight += op.blob.derivative(detectorCoord / scaling);
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
                    total_weight += op.blob.derivative(detectorCoord / scaling);
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
                    total_weight += op.blob.derivative(detectorCoord / scaling);
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
                    total_weight += op.blob.derivative(detectorCoord / scaling);
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
        auto op = PhaseContrastBlobVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;

        WHEN("Applying the PhaseContrastBlobVoxelProjectorCUDA to the volume")
        {
            x = 0;
            // set all voxels on the ray to 1
            x(2, 0) = 1;
            x(2, 1) = 1;
            x(2, 2) = 1;
            x(2, 3) = 1;
            x(2, 4) = 1;

            op.apply(x, Ax);

            const auto weight = op.blob.derivative(0);
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
                    total_weight += op.blob.derivative(detectorCoord / scaling);
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
                    total_weight += op.blob.derivative(detectorCoord / scaling);
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
        auto op = PhaseContrastBlobVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;

        WHEN("Applying the PhaseContrastBlobVoxelProjectorCUDA to the volume")
        {
            // set all voxels on the ray to 1
            x(0, 0) = 1;
            x(1, 1) = 1;
            x(2, 2) = 1;
            x(3, 3) = 1;
            x(4, 4) = 1;

            op.apply(x, Ax);

            const auto weight = op.blob.derivative(0);
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
        auto op = PhaseContrastBlobVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;

        WHEN("Applying the PhaseContrastBlobVoxelProjectorCUDA to the volume")
        {
            x = 0;
            // set all voxels on the ray to 1
            x(0, 2) = 1;
            x(1, 2) = 1;
            x(2, 2) = 1;
            x(3, 2) = 1;
            x(4, 2) = 1;

            op.apply(x, Ax);

            const auto weight = op.blob.derivative(0);
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
        auto op = PhaseContrastBlobVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;

        WHEN("Applying the PhaseContrastBlobVoxelProjectorCUDA to the volume")
        {
            // set all voxels on the ray to 1
            x(0, 4) = 1;
            x(1, 3) = 1;
            x(2, 2) = 1;
            x(3, 1) = 1;
            x(4, 0) = 1;

            op.apply(x, Ax);

            const auto weight = op.blob.derivative(0);
            CAPTURE(weight);

            THEN("Each detector value is equal to 5 * the weight of distance 0")
            {
                CHECK_EQ(Ax[0], Approx(5 * weight));
            }
        }
    }
}

TEST_CASE_TEMPLATE(
    "PhaseContrastBSplineVoxelProjectorCUDA: Testing simple volume 2D with three detector pixels",
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
            auto op = PhaseContrastBSplineVoxelProjectorCUDA<data_t>(domain, range);

            auto Ax = DataContainer<data_t>(range);
            Ax = 0;

            WHEN("only the center voxel is set to 1")
            {
                // set center voxel to 1
                x(2, 2) = 1;

                op.apply(x, Ax);

                const auto weight_dist0 = op.bspline.derivative(0);
                const auto weight_dist1 = op.bspline.derivative(0.95233786);
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
                    CHECK_EQ(-weight_dist1, Approx(Ax[2]).epsilon(0.01));
                }
            }
        }
    }
}

TEST_CASE_TEMPLATE("PhaseContrastBSplineVoxelProjectorCUDA: Test single detector pixel", data_t,
                   float, double)
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
        auto op = PhaseContrastBSplineVoxelProjectorCUDA<data_t>(domain, range);

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

            const auto weight = op.bspline.derivative(0);
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
                    total_weight += op.bspline.derivative(detectorCoord / scaling);
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
                    total_weight += op.bspline.derivative(detectorCoord / scaling);
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
        auto op = PhaseContrastBSplineVoxelProjectorCUDA<data_t>(domain, range);

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

            const auto weight = op.bspline.derivative(0);
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
                    total_weight += op.bspline.derivative(detectorCoord / scaling);
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
                    total_weight += op.bspline.derivative(detectorCoord / scaling);
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
                    total_weight += op.bspline.derivative(detectorCoord / scaling);
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
                    total_weight += op.bspline.derivative(detectorCoord / scaling);
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
        auto op = PhaseContrastBSplineVoxelProjectorCUDA<data_t>(domain, range);

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

            const auto weight = op.bspline.derivative(0);
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
                    total_weight += op.bspline.derivative(detectorCoord / scaling);
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
                    total_weight += op.bspline.derivative(detectorCoord / scaling);
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
        auto op = PhaseContrastBSplineVoxelProjectorCUDA<data_t>(domain, range);

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

            const auto weight = op.bspline.derivative(0);
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
                    total_weight += op.bspline.derivative(detectorCoord / scaling);
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
                    total_weight += op.bspline.derivative(detectorCoord / scaling);
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
                    total_weight += op.bspline.derivative(detectorCoord / scaling);
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
                    total_weight += op.bspline.derivative(detectorCoord / scaling);
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
        auto op = PhaseContrastBSplineVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;

        WHEN("Applying the PhaseContrastBSplineVoxelProjectorCUDA to the volume")
        {
            x = 0;
            // set all voxels on the ray to 1
            x(2, 0) = 1;
            x(2, 1) = 1;
            x(2, 2) = 1;
            x(2, 3) = 1;
            x(2, 4) = 1;

            op.apply(x, Ax);

            const auto weight = op.bspline.derivative(0);
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
                    total_weight += op.bspline.derivative(detectorCoord / scaling);
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
                    total_weight += op.bspline.derivative(detectorCoord / scaling);
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
        auto op = PhaseContrastBSplineVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;

        WHEN("Applying the PhaseContrastBSplineVoxelProjectorCUDA to the volume")
        {
            // set all voxels on the ray to 1
            x(0, 0) = 1;
            x(1, 1) = 1;
            x(2, 2) = 1;
            x(3, 3) = 1;
            x(4, 4) = 1;

            op.apply(x, Ax);

            const auto weight = op.bspline.derivative(0);
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
        auto op = PhaseContrastBSplineVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;

        WHEN("Applying the PhaseContrastBSplineVoxelProjectorCUDA to the volume")
        {
            x = 0;
            // set all voxels on the ray to 1
            x(0, 2) = 1;
            x(1, 2) = 1;
            x(2, 2) = 1;
            x(3, 2) = 1;
            x(4, 2) = 1;

            op.apply(x, Ax);

            const auto weight = op.bspline.derivative(0);
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
        auto op = PhaseContrastBSplineVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;

        WHEN("Applying the PhaseContrastBSplineVoxelProjectorCUDA to the volume")
        {
            // set all voxels on the ray to 1
            x(0, 4) = 1;
            x(1, 3) = 1;
            x(2, 2) = 1;
            x(3, 1) = 1;
            x(4, 0) = 1;

            op.apply(x, Ax);

            const auto weight = op.bspline.derivative(0);
            CAPTURE(weight);

            THEN("Each detector value is equal to 5 * the weight of distance 0")
            {
                CHECK_EQ(Ax[0], Approx(5 * weight));
            }
        }
    }
}

TEST_CASE_TEMPLATE("PhaseContrastBSplineVoxelProjectorCUDA: Test backward projection", data_t,
                   float, double)
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
        auto op = PhaseContrastBSplineVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;
        x = 0;

        WHEN("Backward projecting the operator")
        {
            Ax[0] = 1;

            op.applyAdjoint(Ax, x);

            THEN("The main direction of the ray is equal to the weight")
            {
                const auto weight = op.bspline.derivative(0);
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
                    auto weight = op.bspline.derivative(detectorCoord / scaling);
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
                    auto weight = op.bspline.derivative(detectorCoord / scaling);
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
        auto op = PhaseContrastBSplineVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;
        x = 0;

        WHEN("Backward projecting the operator")
        {
            Ax[0] = 1;

            op.applyAdjoint(Ax, x);

            THEN("The main direction of the ray is equal to the weight")
            {
                const auto weight = op.bspline.derivative(0);
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
                    auto weight = op.bspline.derivative(detectorCoord / scaling);
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
                    auto weight = op.bspline.derivative(detectorCoord / scaling);
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
        auto op = PhaseContrastBSplineVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;
        x = 0;

        WHEN("Backward projecting the operator")
        {
            Ax[0] = 1;

            op.applyAdjoint(Ax, x);

            THEN("The main direction of the ray is equal to the weight")
            {
                const auto weight = op.bspline.derivative(0);
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
                    auto weight = op.bspline.derivative(detectorCoord / scaling);
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
                    auto weight = op.bspline.derivative(detectorCoord / scaling);
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
        auto op = PhaseContrastBSplineVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;
        x = 0;

        WHEN("Backward projecting the operator")
        {
            Ax[0] = 1;

            op.applyAdjoint(Ax, x);

            THEN("The main direction of the ray is equal to the weight")
            {
                const auto weight = op.bspline.derivative(0);
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
                    auto weight = op.bspline.derivative(detectorCoord / scaling);
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
                    auto weight = op.bspline.derivative(detectorCoord / scaling);
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
        auto op = PhaseContrastBSplineVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;
        x = 0;

        WHEN("Backward projecting the operator")
        {
            Ax[0] = 1;

            op.applyAdjoint(Ax, x);

            THEN("The main direction of the ray is equal to the weight")
            {
                const auto weight = op.bspline.derivative(0);
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
                    auto weight = op.bspline.derivative(detectorCoord / scaling);
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
                    auto weight = op.bspline.derivative(detectorCoord / scaling);
                    CHECK_EQ(x(slice + 1, slice), Approx(weight));
                }
            }

            THEN("The pixel 2 above the main direction of the ray have the projected weight")
            {
                for (index_t slice : {0, 1, 2}) {
                    auto sliceF = static_cast<float>(slice);
                    RealVector_t voxCoord{{sliceF, 2 + sliceF}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.bspline.derivative(detectorCoord / scaling);
                    CHECK_EQ(x(slice, 2 + slice), Approx(weight));
                }
            }

            THEN("The pixel 2 above the main direction of the ray have the projected weight")
            {
                for (index_t slice : {0, 1, 2}) {
                    auto sliceF = static_cast<float>(slice);
                    RealVector_t voxCoord{{2 + sliceF, sliceF}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.bspline.derivative(detectorCoord / scaling);
                    CHECK_EQ(x(2 + slice, slice), Approx(weight));
                }
            }
        }
    }
}

TEST_CASE_TEMPLATE("PhaseContrastBlobVoxelProjectorCUDA: Test backward projection", data_t, float,
                   double)
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
        auto op = PhaseContrastBlobVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;
        x = 0;

        WHEN("Backward projecting the operator")
        {
            Ax[0] = 1;

            op.applyAdjoint(Ax, x);

            THEN("The main direction of the ray is equal to the weight")
            {
                const auto weight = op.blob.derivative(0);
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
                    auto weight = op.blob.derivative(detectorCoord / scaling);
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
                    auto weight = op.blob.derivative(detectorCoord / scaling);
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
        auto op = PhaseContrastBlobVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;
        x = 0;

        WHEN("Backward projecting the operator")
        {
            Ax[0] = 1;

            op.applyAdjoint(Ax, x);

            THEN("The main direction of the ray is equal to the weight")
            {
                const auto weight = op.blob.derivative(0);
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
                    auto weight = op.blob.derivative(detectorCoord / scaling);
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
                    auto weight = op.blob.derivative(detectorCoord / scaling);
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
        auto op = PhaseContrastBlobVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;
        x = 0;

        WHEN("Backward projecting the operator")
        {
            Ax[0] = 1;

            op.applyAdjoint(Ax, x);

            THEN("The main direction of the ray is equal to the weight")
            {
                const auto weight = op.blob.derivative(0);
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
                    auto weight = op.blob.derivative(detectorCoord / scaling);
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
                    auto weight = op.blob.derivative(detectorCoord / scaling);
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
        auto op = PhaseContrastBlobVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;
        x = 0;

        WHEN("Backward projecting the operator")
        {
            Ax[0] = 1;

            op.applyAdjoint(Ax, x);

            THEN("The main direction of the ray is equal to the weight")
            {
                const auto weight = op.blob.derivative(0);
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
                    auto weight = op.blob.derivative(detectorCoord / scaling);
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
                    auto weight = op.blob.derivative(detectorCoord / scaling);
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
        auto op = PhaseContrastBlobVoxelProjectorCUDA<data_t>(domain, range);

        auto Ax = DataContainer<data_t>(range);
        Ax = 0;
        x = 0;

        WHEN("Backward projecting the operator")
        {
            Ax[0] = 1;

            op.applyAdjoint(Ax, x);

            THEN("The main direction of the ray is equal to the weight")
            {
                const auto weight = op.blob.derivative(0);
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
                    auto weight = op.blob.derivative(detectorCoord / scaling);
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
                    auto weight = op.blob.derivative(detectorCoord / scaling);
                    CHECK_EQ(x(slice + 1, slice), Approx(weight));
                }
            }

            THEN("The pixel 2 above the main direction of the ray have the projected weight")
            {
                for (index_t slice : {0, 1, 2}) {
                    auto sliceF = static_cast<float>(slice);
                    RealVector_t voxCoord{{sliceF, 2 + sliceF}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.blob.derivative(detectorCoord / scaling);
                    CHECK_EQ(x(slice, 2 + slice), Approx(weight));
                }
            }

            THEN("The pixel 2 above the main direction of the ray have the projected weight")
            {
                for (index_t slice : {0, 1, 2}) {
                    auto sliceF = static_cast<float>(slice);
                    RealVector_t voxCoord{{2 + sliceF, sliceF}};
                    voxCoord = voxCoord.array() + 0.5f;
                    auto [detectorCoordShifted, scaling] =
                        range.projectAndScaleVoxelOnDetector(voxCoord, 0);
                    data_t detectorCoord = detectorCoordShifted[0] - 0.5f;
                    auto weight = op.blob.derivative(detectorCoord / scaling);
                    CHECK_EQ(x(2 + slice, slice), Approx(weight));
                }
            }
        }
    }
}

TEST_CASE_TEMPLATE(
    "PhaseContrastBSplineVoxelProjectorCUDA: Testing simple volume 2D with two detector pixels",
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
            auto op = PhaseContrastBSplineVoxelProjectorCUDA<data_t>(domain, range);

            auto Ax = DataContainer<data_t>(range);
            Ax = 0;

            WHEN("only the center voxel is set to 1")
            {
                // set center voxel to 1
                x(2, 2) = 1;

                op.apply(x, Ax);

                const auto weight_dist_half = op.bspline.derivative(0.4761878);
                CAPTURE(weight_dist_half);

                THEN("Detector values are reverse")
                {
                    CAPTURE(Ax[0]);
                    CAPTURE(Ax[1]);
                    CHECK_EQ(weight_dist_half, Approx(Ax[0]).epsilon(0.01));
                    CHECK_EQ(-weight_dist_half, Approx(Ax[1]).epsilon(0.01));
                }
            }
        }
    }
}

TEST_CASE_TEMPLATE("PhaseContrastBlobVoxelProjectorCUDA: Testing simple volume 2D with three "
                   "detector pixels (left/right pixel)",
                   data_t, float, double)
{
    for (index_t size = 128; size <= 256; size *= 2) {
        const IndexVector_t sizeDomain({{size, size}});
        const IndexVector_t sizeRange({{3, 1}});
        auto middlePixel = 1;

        auto domain = VolumeDescriptor(sizeDomain);
        auto x = DataContainer<data_t>(domain);
        x = 0;

        auto stc = SourceToCenterOfRotation{static_cast<real_t>(10000000 * size)};
        auto ctr = CenterOfRotationToDetector{static_cast<real_t>(size)};
        auto volData = VolumeData2D{Size2D{sizeDomain}};
        auto sinoData = SinogramData2D{Size2D{sizeRange}};

        GIVEN("a volume " + std::to_string(size) + "^2, at 45 degree")
        {
            std::vector<Geometry> geom;
            geom.emplace_back(stc, ctr, Degree{45}, VolumeData2D{Size2D{sizeDomain}},
                              SinogramData2D{Size2D{sizeRange}});

            auto range = PlanarDetectorDescriptor(sizeRange, geom);
            auto op = PhaseContrastBlobVoxelProjectorCUDA<data_t>(domain, range);

            auto Ax = DataContainer<data_t>(range);
            Ax = 0;

            WHEN("Setting only the voxels directly on the middle ray to 1")
            {
                // set all voxels on the ray to 1
                for (int i = 0; i < size; i++) {
                    x(i, i) = 1;
                }

                op.apply(x, Ax);

                THEN("The the detector value is equal to size * the weight of distance sqrt(0.5)")
                {
                    data_t total_weight =
                        (size) *op.blob.derivative(middlePixel + distance_from_center(0));
                    CHECK_EQ(Ax[0], Approx(total_weight));
                    CHECK_EQ(Ax[2], Approx(-total_weight));
                }
            }

            WHEN("Setting only the voxels directly below the middle ray to 1")
            {
                x = 0;
                for (int i = 0; i < size - 1; i++) {
                    x(i, i + 1) = 1;
                }

                op.apply(x, Ax);

                THEN("The detector value is equal to the weight at the scaled distances")
                {
                    data_t total_weight =
                        (size - 1) * op.blob.derivative(middlePixel + distance_from_center(-1));
                    CHECK_EQ(Ax[0], Approx(total_weight));
                    total_weight =
                        (size - 1) * op.blob.derivative(-middlePixel + distance_from_center(-1));
                    CHECK_EQ(Ax[2], Approx(total_weight));
                }
            }

            WHEN("Setting only the voxels directly above the ray to 1")
            {
                x = 0;
                for (int i = 0; i < size - 1; i++) {
                    x(i + 1, i) = 1;
                }

                op.apply(x, Ax);

                THEN("The detector value is equal to the weight at the scaled distances")
                {
                    data_t total_weight =
                        (size - 1) * op.blob.derivative(middlePixel + distance_from_center(1));
                    CHECK_EQ(Ax[0], Approx(total_weight));
                    total_weight =
                        (size - 1) * op.blob.derivative(-middlePixel + distance_from_center(1));
                    CHECK_EQ(Ax[2], Approx(total_weight));
                }
            }

            WHEN("Setting only the voxels two steps below the ray")
            {
                x = 0;
                for (int i = 0; i < size - 2; i++) {
                    x(i, i + 2) = 1;
                }

                op.apply(x, Ax);

                THEN("The detector value is equal to the weight at the scaled distances")
                {
                    data_t total_weight =
                        (size - 2) * op.blob.derivative(middlePixel + distance_from_center(-2));
                    CHECK_EQ(Ax[0], Approx(total_weight));
                    total_weight =
                        (size - 2) * op.blob.derivative(-middlePixel + distance_from_center(-2));
                    CHECK_EQ(Ax[2], Approx(total_weight));
                }
            }

            WHEN("Setting only the voxels two steps above the ray")
            {
                x = 0;
                for (int i = 0; i < size - 2; i++) {
                    x(i + 2, i) = 1;
                }

                op.apply(x, Ax);

                THEN("The detector value is equal to the weight at the scaled distances")
                {
                    data_t total_weight =
                        (size - 2) * op.blob.derivative(middlePixel + distance_from_center(2));
                    CHECK_EQ(Ax[0], Approx(total_weight));
                    total_weight =
                        (size - 2) * op.blob.derivative(-middlePixel + distance_from_center(2));
                    CHECK_EQ(Ax[2], Approx(total_weight));
                }
            }

            WHEN("Setting the three middle rays to 1")
            {
                // set all voxels on the ray to 1
                x = 0;
                for (int i = 0; i < size - 1; i++) {
                    x(i, i) = 1;
                    x(i, i + 1) = 1;
                    x(i + 1, i) = 1;
                }
                x(size - 1, size - 1) = 1;

                op.apply(x, Ax);

                THEN("The detector value is equal to size * the weight of distance 0")
                {
                    // relative positions of the rays 1 - pos, 1, 1 + pos
                    data_t total_weight =
                        (size - 1) * op.blob.derivative(middlePixel + distance_from_center(-1))
                        + size * op.blob.derivative(middlePixel + distance_from_center(0))
                        + (size - 1) * op.blob.derivative(middlePixel + distance_from_center(1));
                    CHECK_EQ(Ax[0], Approx(total_weight).epsilon(std::abs(total_weight / 1000.0)));
                    CHECK_EQ(Ax[2], Approx(-total_weight).epsilon(std::abs(total_weight / 1000.0)));
                }
            }

            WHEN("Setting the five middle rays to 1")
            {
                // set all voxels on the ray to 1
                x = 0;
                for (int i = 0; i < size - 1; i++) {
                    x(i, i) = 1;
                    x(i, i + 1) = 1;
                    x(i + 1, i) = 1;
                }
                for (int i = 0; i < size - 2; i++) {
                    x(i, i + 2) = 1;
                    x(i + 2, i) = 1;
                }
                x(size - 1, size - 1) = 1;

                op.apply(x, Ax);

                THEN("The detector value is equal to size * the weight of distance 0")
                {
                    // relative positions of the rays  1 - posB, 1 - posA, 1, 1 + posA, (1 + posB)
                    data_t total_weight =
                        (size - 2) * op.blob.derivative(middlePixel + distance_from_center(-2))
                        + (size - 1) * op.blob.derivative(middlePixel + distance_from_center(-1))
                        + size * op.blob.derivative(middlePixel + distance_from_center(0))
                        + (size - 1) * op.blob.derivative(middlePixel + distance_from_center(1))
                        + (size - 2) * op.blob.derivative(middlePixel + distance_from_center(2));
                    CHECK_EQ(Ax[0], Approx(total_weight));
                    CHECK_EQ(Ax[2], Approx(-total_weight));
                }
            }

            WHEN("Setting the 9 middle rays to 1")
            {
                // set all voxels on the ray to 1
                x = 0;
                for (int i = 0; i < size; i++) {
                    x(i, i) = 1;
                }
                for (int i = 0; i < size - 1; i++) {
                    x(i, i + 1) = 1;
                    x(i + 1, i) = 1;
                }
                for (int i = 0; i < size - 2; i++) {
                    x(i, i + 2) = 1;
                    x(i + 2, i) = 1;
                }
                for (int i = 0; i < size - 3; i++) {
                    x(i, i + 3) = 1;
                    x(i + 3, i) = 1;
                }
                for (int i = 0; i < size - 4; i++) {
                    x(i, i + 4) = 1;
                    x(i + 4, i) = 1;
                }

                op.apply(x, Ax);

                THEN("The detector value is equal to size * the weight of distance 0")
                {
                    // relative positions of the rays  1 - posB, 1 - posA, 1, 1 + posA, (1 + posB)
                    data_t total_weight =
                        (size - 4) * op.blob.derivative(middlePixel + distance_from_center(-4))
                        + (size - 3) * op.blob.derivative(middlePixel + distance_from_center(-3))
                        + (size - 2) * op.blob.derivative(middlePixel + distance_from_center(-2))
                        + (size - 1) * op.blob.derivative(middlePixel + distance_from_center(-1))
                        + size * op.blob.derivative(middlePixel + distance_from_center(0))
                        + (size - 1) * op.blob.derivative(middlePixel + distance_from_center(1))
                        + (size - 2) * op.blob.derivative(middlePixel + distance_from_center(2))
                        + (size - 3) * op.blob.derivative(middlePixel + distance_from_center(3))
                        + (size - 4) * op.blob.derivative(middlePixel + distance_from_center(4));
                    CHECK_EQ(Ax[0], Approx(total_weight).epsilon(0.001));
                    CHECK_EQ(Ax[2], Approx(-total_weight).epsilon(0.001));
                }
            }

            WHEN("Setting all the voxels along a slice to a distribution")
            {
                // set all voxels on the ray to 1
                x = 0;
                for (int i = 0; i < size; i++) {
                    x(i, i) = 1;
                }
                for (int i = 0; i < size - 1; i++) {
                    x(i, i + 1) = .9;
                    x(i + 1, i) = .9;
                }
                for (int i = 0; i < size - 2; i++) {
                    x(i, i + 2) = .8;
                    x(i + 2, i) = .8;
                }
                for (int i = 0; i < size - 3; i++) {
                    x(i, i + 3) = .7;
                    x(i + 3, i) = .7;
                }
                for (int i = 0; i < size - 4; i++) {
                    x(i, i + 4) = .6;
                    x(i + 4, i) = .6;
                }

                op.apply(x, Ax);

                THEN("The detector value is equal to size * the weight of distance 0")
                {
                    // relative positions of the rays  1 - posB, 1 - posA, 1, 1 + posA, (1 + posB)
                    data_t total_weight =
                        .6 * (size - 4) * op.blob.derivative(middlePixel + distance_from_center(-4))
                        + .7 * (size - 3)
                              * op.blob.derivative(middlePixel + distance_from_center(-3))
                        + .8 * (size - 2)
                              * op.blob.derivative(middlePixel + distance_from_center(-2))
                        + .9 * (size - 1)
                              * op.blob.derivative(middlePixel + distance_from_center(-1))
                        + size * op.blob.derivative(middlePixel + distance_from_center(0))
                        + .9 * (size - 1)
                              * op.blob.derivative(middlePixel + distance_from_center(1))
                        + .8 * (size - 2)
                              * op.blob.derivative(middlePixel + distance_from_center(2))
                        + .7 * (size - 3)
                              * op.blob.derivative(middlePixel + distance_from_center(3))
                        + .6 * (size - 4)
                              * op.blob.derivative(middlePixel + distance_from_center(4));
                    CHECK_EQ(Ax[0], Approx(total_weight).epsilon(0.001));
                    CHECK_EQ(Ax[2], Approx(-total_weight).epsilon(0.001));
                }
            }
            WHEN("Setting the voxels to 1")
            {
                // set all voxels on the ray to 1
                x = 1;

                op.apply(x, Ax);

                THEN("The detector value is equal to size * the weight of distance 0")
                {
                    // relative positions of the rays  1 - posB, 1 - posA, 1, 1 + posA, (1 + posB)
                    data_t total_weight =
                        (size - 4) * op.blob.derivative(middlePixel + distance_from_center(-4))
                        + (size - 3) * op.blob.derivative(middlePixel + distance_from_center(-3))
                        + (size - 2) * op.blob.derivative(middlePixel + distance_from_center(-2))
                        + (size - 1) * op.blob.derivative(middlePixel + distance_from_center(-1))
                        + size * op.blob.derivative(middlePixel + distance_from_center(0))
                        + (size - 1) * op.blob.derivative(middlePixel + distance_from_center(1))
                        + (size - 2) * op.blob.derivative(middlePixel + distance_from_center(2))
                        + (size - 3) * op.blob.derivative(middlePixel + distance_from_center(3))
                        + (size - 4) * op.blob.derivative(middlePixel + distance_from_center(4));
                    CHECK_EQ(Ax[0], Approx(total_weight).epsilon(0.001));
                    CHECK_EQ(Ax[2], Approx(-total_weight).epsilon(0.001));
                }
            }

            WHEN("Using a smoothBlob")
            {
                // set all voxels on the ray to 1
                x = 0;

                auto radius = size / 2;

                for (index_t i = 0; i < x.getSize(); i++) {
                    const RealVector_t coord = x.getDataDescriptor()
                                                   .getCoordinateFromIndex(i)
                                                   .template cast<real_t>()
                                                   .array()
                                               + 0.5;
                    data_t distance_from_center =
                        (coord - x.getDataDescriptor().getLocationOfOrigin()).norm();
                    x[i] = blobs::blob_evaluate(distance_from_center, radius, 10.83f, 2);
                }

                op.apply(x, Ax);

                THEN("The detector value is equal to size * the weight of distance 0")
                {
                    // relative positions of the rays  1 - posB, 1 - posA, 1, 1 + posA, (1 + posB)
                    data_t total_weight = 0;
                    for (index_t i = 0; i < size; i++) {
                        total_weight +=
                            x(i, i) * op.blob.derivative(middlePixel + distance_from_center(0));
                    }
                    for (index_t i = 0; i < size - 1; i++) {
                        total_weight +=
                            x(i + 1, i) * op.blob.derivative(middlePixel + distance_from_center(1));
                    }
                    for (index_t i = 0; i < size - 1; i++) {
                        total_weight +=
                            x(i, i + 1)
                            * op.blob.derivative(middlePixel + distance_from_center(-1));
                    }
                    for (index_t i = 0; i < size - 2; i++) {
                        total_weight +=
                            x(i + 2, i) * op.blob.derivative(middlePixel + distance_from_center(2));
                    }
                    for (index_t i = 0; i < size - 2; i++) {
                        total_weight +=
                            x(i, i + 2)
                            * op.blob.derivative(middlePixel + distance_from_center(-2));
                    }
                    for (index_t i = 0; i < size - 3; i++) {
                        total_weight +=
                            x(i + 3, i) * op.blob.derivative(middlePixel + distance_from_center(3));
                    }
                    for (index_t i = 0; i < size - 3; i++) {
                        total_weight +=
                            x(i, i + 3)
                            * op.blob.derivative(middlePixel + distance_from_center(-3));
                    }
                    for (index_t i = 0; i < size - 4; i++) {
                        total_weight +=
                            x(i + 4, i) * op.blob.derivative(middlePixel + distance_from_center(4));
                    }
                    for (index_t i = 0; i < size - 4; i++) {
                        total_weight +=
                            x(i, i + 4)
                            * op.blob.derivative(middlePixel + distance_from_center(-4));
                    }
                    CHECK_EQ(Ax[0], Approx(total_weight).epsilon(0.001));
                    CHECK_EQ(Ax[2], Approx(-total_weight).epsilon(0.001));
                }
            }
            WHEN("Using a smoothBlob")
            {
                // set all voxels on the ray to 1
                x = 0;

                auto radius = size / 2;

                for (index_t i = 0; i < x.getSize(); i++) {
                    const RealVector_t coord = x.getDataDescriptor()
                                                   .getCoordinateFromIndex(i)
                                                   .template cast<real_t>()
                                                   .array()
                                               + 0.5;
                    data_t distance_from_center =
                        (coord - x.getDataDescriptor().getLocationOfOrigin()).norm();
                    x[i] = blobs::blob_evaluate(distance_from_center, radius, 10.83f, 2);
                }
                for (int angle = 0; angle < 360; angle += 6) {
                    GIVEN("a volume " + std::to_string(size) + "^2, at " + std::to_string(angle)
                          + " degree")
                    {
                        std::vector<Geometry> geom;
                        geom.emplace_back(stc, ctr, Degree{static_cast<real_t>(angle)},
                                          VolumeData2D{Size2D{sizeDomain}},
                                          SinogramData2D{Size2D{sizeRange}});

                        auto range = PlanarDetectorDescriptor(sizeRange, geom);
                        auto op = PhaseContrastBlobVoxelProjectorCUDA<data_t>(domain, range);

                        auto Ax = DataContainer<data_t>(range);
                        Ax = 0;

                        auto radians = angle * M_PI / 180.0;

                        data_t total_weight = 0;
                        RealVector_t middle =
                            x.getDataDescriptor().getLocationOfOrigin().array() - 0.5;
                        for (index_t i = 0; i < x.getSize(); i++) {
                            const IndexVector_t coordIndex =
                                x.getDataDescriptor().getCoordinateFromIndex(i);
                            const RealVector_t coord =
                                coordIndex.template cast<real_t>().array() + 0.5;

                            total_weight += x[i]
                                            * op.blob.derivative(middlePixel
                                                                 + distance_from_center(
                                                                     coordIndex, middle, radians));
                            data_t distance_from_center =
                                (coord - x.getDataDescriptor().getLocationOfOrigin()).norm();
                        }

                        op.apply(x, Ax);

                        THEN("The detector value is equal to the parallel projected voxel")
                        {
                            // relative positions of the rays  1 - posB, 1 - posA, 1, 1 + posA, (1 +
                            // posB)

                            CHECK_EQ(Ax[0], Approx(total_weight).epsilon(0.001));
                            CHECK_EQ(Ax[2], Approx(-total_weight).epsilon(0.001));
                        }
                    }
                }
            }
        }
    }
}

TEST_SUITE_END();
