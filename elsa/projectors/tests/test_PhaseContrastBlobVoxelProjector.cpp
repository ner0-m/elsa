#include "doctest/doctest.h"

#include "PhaseContrastProjector.h"
#include "PlanarDetectorDescriptor.h"

#include "PrettyPrint/Eigen.h"
#include "PrettyPrint/Stl.h"
#include "spdlog/fmt/bundled/core.h"

using namespace elsa;
using namespace elsa::geometry;
using namespace doctest;

TEST_SUITE_BEGIN("phase contrast blob projector");

Eigen::IOFormat vecfmt(10, 0, ", ", ", ", "", "", "[", "]");
Eigen::IOFormat matfmt(10, 0, ", ", "\n", "\t\t[", "]");

TYPE_TO_STRING(PhaseContrastBlobVoxelProjector<float>);

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

TEST_CASE_TEMPLATE("PhaseContrastBlobVoxelProjector: Testing simple volume 2D with three detector "
                   "pixels (left/right pixel)",
                   data_t, float, double)
{
    for (index_t size = 8; size <= 256; size *= 2) {
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
            auto op = PhaseContrastBlobVoxelProjector<data_t>(domain, range);
            const auto weight = [op](data_t s) {
                return op.blob.get_derivative_lut()(std::abs(s)) * math::sgn(s);
            };

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
                    data_t total_weight = (size) *weight(middlePixel + distance_from_center(0));
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
                        (size - 1) * weight(middlePixel + distance_from_center(-1));
                    CHECK_EQ(Ax[0], Approx(total_weight).epsilon(0.001));
                    total_weight = (size - 1) * weight(-middlePixel + distance_from_center(-1));
                    CHECK_EQ(Ax[2], Approx(total_weight).epsilon(0.001));
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
                        (size - 1) * weight(middlePixel + distance_from_center(1));
                    CHECK_EQ(Ax[0], Approx(total_weight).epsilon(0.001));
                    total_weight = (size - 1) * weight(-middlePixel + distance_from_center(1));
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
                        (size - 2) * weight(middlePixel + distance_from_center(-2));
                    CHECK_EQ(Ax[0], Approx(total_weight));
                    total_weight = (size - 2) * weight(-middlePixel + distance_from_center(-2));
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
                        (size - 2) * weight(middlePixel + distance_from_center(2));
                    CHECK_EQ(Ax[0], Approx(total_weight));
                    total_weight = (size - 2) * weight(-middlePixel + distance_from_center(2));
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
                        (size - 1) * weight(middlePixel + distance_from_center(-1))
                        + size * weight(middlePixel + distance_from_center(0))
                        + (size - 1) * weight(middlePixel + distance_from_center(1));
                    CHECK_EQ(Ax[0], Approx(total_weight));
                    CHECK_EQ(Ax[2], Approx(-total_weight));
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
                        (size - 2) * weight(middlePixel + distance_from_center(-2))
                        + (size - 1) * weight(middlePixel + distance_from_center(-1))
                        + size * weight(middlePixel + distance_from_center(0))
                        + (size - 1) * weight(middlePixel + distance_from_center(1))
                        + (size - 2) * weight(middlePixel + distance_from_center(2));
                    CHECK_EQ(Ax[0], Approx(total_weight).epsilon(-0.0001 * total_weight));
                    CHECK_EQ(Ax[2], Approx(-total_weight).epsilon(-0.0001 * total_weight));
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
                        (size - 4) * weight(middlePixel + distance_from_center(-4))
                        + (size - 3) * weight(middlePixel + distance_from_center(-3))
                        + (size - 2) * weight(middlePixel + distance_from_center(-2))
                        + (size - 1) * weight(middlePixel + distance_from_center(-1))
                        + size * weight(middlePixel + distance_from_center(0))
                        + (size - 1) * weight(middlePixel + distance_from_center(1))
                        + (size - 2) * weight(middlePixel + distance_from_center(2))
                        + (size - 3) * weight(middlePixel + distance_from_center(3))
                        + (size - 4) * weight(middlePixel + distance_from_center(4));
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
                        .6 * (size - 4) * weight(middlePixel + distance_from_center(-4))
                        + .7 * (size - 3) * weight(middlePixel + distance_from_center(-3))
                        + .8 * (size - 2) * weight(middlePixel + distance_from_center(-2))
                        + .9 * (size - 1) * weight(middlePixel + distance_from_center(-1))
                        + size * weight(middlePixel + distance_from_center(0))
                        + .9 * (size - 1) * weight(middlePixel + distance_from_center(1))
                        + .8 * (size - 2) * weight(middlePixel + distance_from_center(2))
                        + .7 * (size - 3) * weight(middlePixel + distance_from_center(3))
                        + .6 * (size - 4) * weight(middlePixel + distance_from_center(4));
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
                        (size - 4) * weight(middlePixel + distance_from_center(-4))
                        + (size - 3) * weight(middlePixel + distance_from_center(-3))
                        + (size - 2) * weight(middlePixel + distance_from_center(-2))
                        + (size - 1) * weight(middlePixel + distance_from_center(-1))
                        + size * weight(middlePixel + distance_from_center(0))
                        + (size - 1) * weight(middlePixel + distance_from_center(1))
                        + (size - 2) * weight(middlePixel + distance_from_center(2))
                        + (size - 3) * weight(middlePixel + distance_from_center(3))
                        + (size - 4) * weight(middlePixel + distance_from_center(4));
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
                        total_weight += x(i, i) * weight(middlePixel + distance_from_center(0));
                    }
                    for (index_t i = 0; i < size - 1; i++) {
                        total_weight += x(i + 1, i) * weight(middlePixel + distance_from_center(1));
                    }
                    for (index_t i = 0; i < size - 1; i++) {
                        total_weight +=
                            x(i, i + 1) * weight(middlePixel + distance_from_center(-1));
                    }
                    for (index_t i = 0; i < size - 2; i++) {
                        total_weight += x(i + 2, i) * weight(middlePixel + distance_from_center(2));
                    }
                    for (index_t i = 0; i < size - 2; i++) {
                        total_weight +=
                            x(i, i + 2) * weight(middlePixel + distance_from_center(-2));
                    }
                    for (index_t i = 0; i < size - 3; i++) {
                        total_weight += x(i + 3, i) * weight(middlePixel + distance_from_center(3));
                    }
                    for (index_t i = 0; i < size - 3; i++) {
                        total_weight +=
                            x(i, i + 3) * weight(middlePixel + distance_from_center(-3));
                    }
                    for (index_t i = 0; i < size - 4; i++) {
                        total_weight += x(i + 4, i) * weight(middlePixel + distance_from_center(4));
                    }
                    for (index_t i = 0; i < size - 4; i++) {
                        total_weight +=
                            x(i, i + 4) * weight(middlePixel + distance_from_center(-4));
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
                for (int angle = 0; angle < 360; angle += 15) {
                    GIVEN("a volume " + std::to_string(size) + "^2, at " + std::to_string(angle)
                          + " degree")
                    {
                        std::vector<Geometry> geom;
                        geom.emplace_back(stc, ctr, Degree{static_cast<real_t>(angle)},
                                          VolumeData2D{Size2D{sizeDomain}},
                                          SinogramData2D{Size2D{sizeRange}});

                        auto range = PlanarDetectorDescriptor(sizeRange, geom);
                        auto op = PhaseContrastBlobVoxelProjector<data_t>(domain, range);

                        auto Ax = DataContainer<data_t>(range);
                        Ax = 0;

                        auto radians = angle * M_PI / 180.0;

                        op.apply(x, Ax);

                        THEN("The detector value is equal to the parallel projected voxel")
                        {
                            // relative positions of the rays  1 - posB, 1 - posA, 1, 1 + posA, (1 +
                            // posB)
                            data_t total_weight = 0;
                            RealVector_t middle =
                                x.getDataDescriptor().getLocationOfOrigin().array() - 0.5;
                            for (index_t i = 0; i < x.getSize(); i++) {
                                const IndexVector_t coordIndex =
                                    x.getDataDescriptor().getCoordinateFromIndex(i);

                                total_weight +=
                                    x[i]
                                    * weight(middlePixel
                                             + distance_from_center(coordIndex, middle, radians));
                            }
                            CHECK_EQ(Ax[0], Approx(total_weight).epsilon(0.001));
                            CHECK_EQ(Ax[2], Approx(-total_weight).epsilon(0.001));
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE_TEMPLATE("PhaseContrastBlobVoxelProjector: Testing simple volume 2D with three detector "
                   "pixels (middle pixel)",
                   data_t, float, double)
{
    for (index_t size = 8; size <= 256; size *= 2) {
        const IndexVector_t sizeDomain({{size, size}});
        const IndexVector_t sizeRange({{3, 1}});
        index_t middlePixel = 1;

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
            auto op = PhaseContrastBlobVoxelProjector<data_t>(domain, range);

            auto Ax = DataContainer<data_t>(range);
            Ax = 0;

            WHEN("Setting only the voxels directly on the ray to 1")
            {
                // set all voxels on the ray to 1
                for (int i = 0; i < size; i++) {
                    x(i, i) = 1;
                }

                op.apply(x, Ax);

                const auto weight = op.blob.derivative(distance_from_center(0));
                CAPTURE(weight);

                THEN("The middle detector value is equal to size * the weight of distance 0")
                {
                    CHECK_EQ(Ax[1], Approx(size * weight).epsilon(0.005));
                }
            }

            WHEN("Setting only the voxels directly below the ray to 1")
            {
                x = 0;
                for (int i = 0; i < size - 1; i++) {
                    x(i, i + 1) = 1;
                }

                op.apply(x, Ax);

                THEN("The detector value is equal to the weight at the scaled distances")
                {
                    data_t total_weight = (size - 1) * op.blob.derivative(distance_from_center(-1));

                    CHECK_EQ(Ax[1], Approx(total_weight));
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
                    data_t total_weight = (size - 1) * op.blob.derivative(distance_from_center(1));

                    CHECK_EQ(Ax[1], Approx(total_weight));
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
                    data_t total_weight = (size - 2) * op.blob.derivative(distance_from_center(-2));

                    CHECK_EQ(Ax[1], Approx(total_weight).epsilon(0.001 * total_weight));
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
                    data_t total_weight = (size - 2) * op.blob.derivative(distance_from_center(2));

                    CHECK_EQ(Ax[1], Approx(total_weight).epsilon(0.001));
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

                THEN("The middle detector value is equal to size * the weight of distance 0")
                {
                    data_t total_weight =
                        (size - 1) * op.blob.derivative(distance_from_center(1))
                        + (size - 1) * op.blob.derivative(distance_from_center(-1))
                        + size * op.blob.derivative(0);
                    CHECK_EQ(Ax[1], Approx(total_weight).epsilon(0.002));
                }
            }
        }
    }
}

TEST_SUITE_END();
