#include "doctest/doctest.h"

#include "LutProjector.h"
#include "PlanarDetectorDescriptor.h"

#include "PrettyPrint/Eigen.h"
#include "PrettyPrint/Stl.h"
#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ostr.h>

using namespace elsa;
using namespace elsa::geometry;
using namespace doctest;

TEST_SUITE_BEGIN("projectors");

Eigen::IOFormat vecfmt(10, 0, ", ", ", ", "", "", "[", "]");
Eigen::IOFormat matfmt(10, 0, ", ", "\n", "\t\t[", "]");

// when one uses fmt 8.1.1 bundled with spdlog 1.10.0, ostream_formatter is not available.
// thus we can only create the ostream_formatter instance if fmt is not bundled.
// but apparently the older fmt does not requires this fmt::formatter definition to use iostreams at
// all, so we can just continue without it.
#if FMT_VERSION > 90000
// https://fmt.dev/latest/api.html#ostream-api
// allow eigen WithFormat things to be ostream-formatted.
template <typename... C>
struct fmt::formatter<Eigen::WithFormat<C...>> : ostream_formatter {
};
#endif

TYPE_TO_STRING(BlobProjector<float>);

// Redefine GIVEN such that it's nicely usable inside an loop
#undef GIVEN
#define GIVEN(...) DOCTEST_SUBCASE((std::string("   Given: ") + std::string(__VA_ARGS__)).c_str())

TEST_CASE_TEMPLATE("BlobProjector: Testing rays going through the center of the volume", data_t,
                   float, double)
{
    // const IndexVector_t sizeDomain({{5, 5, 5}});
    // const IndexVector_t sizeRange({{1, 1, 1}});
    const IndexVector_t sizeDomain({{1, 1}});
    const IndexVector_t sizeRange({{1, 1}});

    auto domain = VolumeDescriptor(sizeDomain);
    auto x = DataContainer<data_t>(domain);
    x = 0;

    auto stc = SourceToCenterOfRotation{100};
    auto ctr = CenterOfRotationToDetector{0};

    const auto theta = 0;
    const auto thetad = Degree{theta}.to_radian();
    // const auto beta = 0;
    // const auto betad = Degree{beta}.to_radian();
    // const auto alpha = 0;
    // const auto alphad = Degree{alpha}.to_radian();
    std::vector<Geometry> geom;
    // geom.emplace_back(stc, ctr, VolumeData3D{Size3D{sizeDomain}},
    // SinogramData3D{Size3D{sizeRange}},
    //                   RotationAngles3D{Gamma{theta}, Beta{beta}, Alpha{alpha}});
    geom.emplace_back(stc, ctr, Degree{theta}, VolumeData2D{Size2D{sizeDomain}},
                      SinogramData2D{Size2D{sizeRange}}, PrincipalPointOffset{0},
                      RotationOffset2D{-0.5, -0.5});
    auto range = PlanarDetectorDescriptor(sizeRange, geom);

    Geometry& g = geom[0];

    auto projInvMatrix = g.getInverseProjectionMatrix();

    MESSAGE(fmt::format("camera center: {}", g.getCameraCenter().format(vecfmt)));
    MESSAGE(fmt::format("Projection Matrix:\n{}", g.getProjectionMatrix().format(matfmt)));

    const RealVector_t principalpoint =
        (projInvMatrix * RealVector_t({{0.5, 1}})).head(2).normalized();
    MESSAGE(fmt::format("principal point: {}", principalpoint.format(vecfmt)));

    const auto projMatrix = g.getProjectionMatrix();
    const auto proj = RealMatrix_t({{std::cos(thetad), std::sin(thetad)}});
    MESSAGE(fmt::format("My Projection Matrix: {}", proj.format(vecfmt)));

    // const auto xaxis = RealMatrix_t({{1, 0, 0}});
    // const auto yaxis = RealMatrix_t({{0, 1, 0}});
    // const auto zaxis = RealMatrix_t({{0, 0, 1}});
    // const auto xaxish = RealMatrix_t({{1, 0, 0, 1}});
    // const auto yaxish = RealMatrix_t({{0, 1, 0, 1}});
    // const auto zaxish = RealMatrix_t({{0, 0, 1, 1}});

    const auto xaxis = RealVector_t({{1, 0}});
    const auto yaxis = RealVector_t({{0, 1}});
    const auto xaxish = RealVector_t({{1, 0, 1}});
    const auto yaxish = RealVector_t({{0, 1, 1}});

    const RealVector_t projx = (projMatrix.block(0, 0, 2, 2) * xaxis).head(1);
    const RealVector_t projy = (projMatrix.block(0, 0, 2, 2) * yaxis).head(1);
    // const RealVector_t projx = (projMatrix * xaxish).hnormalized();
    // const RealVector_t projy = (projMatrix * yaxish).hnormalized();
    // const RealVector_t projz = (projMatrix * zaxish);

    MESSAGE(fmt::format("project xaxis: {}", projx.format(vecfmt)));
    MESSAGE(fmt::format("project yaxis: {}", projy.format(vecfmt)));

    const RealVector_t myprojx = proj * xaxis;
    const RealVector_t myprojy = proj * yaxis;
    MESSAGE(fmt::format("project xaxis: {}", myprojx.format(vecfmt)));
    MESSAGE(fmt::format("project yaxis: {}", myprojy.format(vecfmt)));
}

TEST_CASE_TEMPLATE("BlobProjector: Testing rays going through the center of the volume", data_t,
                   float, double)
{
    const IndexVector_t sizeDomain({{5, 5, 5}});
    const IndexVector_t sizeRange({{1, 1, 1}});

    auto domain = VolumeDescriptor(sizeDomain);
    auto x = DataContainer<data_t>(domain);
    x = 0;

    auto stc = SourceToCenterOfRotation{100};
    auto ctr = CenterOfRotationToDetector{5};

    // set center voxel to 1
    x(2, 2, 2) = 1;

    for (int i = 0; i < 360; i += 4) {
        GIVEN("Ray of angle " + std::to_string(i))
        {
            std::vector<Geometry> geom;
            geom.emplace_back(stc, ctr, VolumeData3D{Size3D{sizeDomain}},
                              SinogramData3D{Size3D{sizeRange}},
                              RotationAngles3D{Gamma{static_cast<real_t>(i)}});
            auto range = PlanarDetectorDescriptor(sizeRange, geom);
            auto op = BlobProjector<data_t>(domain, range);

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
                    CHECK_EQ(weight, Approx(Ax[0]));
                }
            }
        }
    }
}

TEST_CASE_TEMPLATE("BlobProjector: Testing rays going through the center of the volume", data_t,
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

    for (int i = 0; i < 360; i += 4) {
        GIVEN("Ray of angle " + std::to_string(i))
        {
            std::vector<Geometry> geom;
            geom.emplace_back(stc, ctr, Degree{static_cast<real_t>(i)},
                              VolumeData2D{Size2D{sizeDomain}}, SinogramData2D{Size2D{sizeRange}});
            auto range = PlanarDetectorDescriptor(sizeRange, geom);
            auto op = BlobProjector<data_t>(domain, range);

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

TEST_CASE_TEMPLATE("BlobProjector: Testing 2 rays going through the center of the volume", data_t,
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
            auto op = BlobProjector<data_t>(domain, range);

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

TEST_CASE_TEMPLATE("BlobProjector: Testing 3 rays going through the center of the volume", data_t,
                   float, double)
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
            auto op = BlobProjector<data_t>(domain, range);

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

TEST_CASE("BlobProjector: Test stepping of single rays")
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
        auto op = BlobProjector(domain, range);

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

            const auto weight = op.weight(1);
            CAPTURE(weight);

            THEN("Each detector value is equal to 5 * the weight of distance 1")
            {
                CHECK_EQ(Ax[0], Approx(5 * weight));
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

            const auto weight = op.weight(1);
            CAPTURE(weight);

            THEN("Each detector value is equal to 5 * the weight of distance 1")
            {
                CHECK_EQ(Ax[0], Approx(5 * weight));
            }
        }

        WHEN("Setting only the voxels 2 away neighbours above the ray to 1")
        {
            // set all voxels above the ray to 1, i.e the visited neighbours
            x(4, 0) = 1;
            x(4, 1) = 1;
            x(4, 2) = 1;
            x(4, 3) = 1;
            x(4, 4) = 1;

            op.apply(x, Ax);

            const auto weight = op.weight(2);
            CAPTURE(weight);

            THEN("Each detector value is equal to 5 * the weight of distance 2")
            {
                CHECK_EQ(Ax[0], Approx(5 * weight));
            }
        }

        WHEN("Setting only the voxels 2 away neighbours below the ray to 1")
        {
            // set all voxels above the ray to 1, i.e the visited neighbours
            x(0, 0) = 1;
            x(0, 1) = 1;
            x(0, 2) = 1;
            x(0, 3) = 1;
            x(0, 4) = 1;

            op.apply(x, Ax);

            const auto weight = op.weight(2);
            CAPTURE(weight);

            THEN("Each detector value is equal to 5 * the weight of distance 2")
            {
                CHECK_EQ(Ax[0], Approx(5 * weight));
            }
        }
    }

    GIVEN("a single detector of size 1, at 45 degree")
    {
        std::vector<Geometry> geom;
        geom.emplace_back(stc, ctr, Degree{45}, VolumeData2D{Size2D{sizeDomain}},
                          SinogramData2D{Size2D{sizeRange}});

        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BlobProjector(domain, range);

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

        WHEN("Setting only the voxels directly above on the ray to 1")
        {
            // set all voxels on the ray to 1
            x(0, 1) = 1;
            x(1, 2) = 1;
            x(2, 3) = 1;
            x(3, 4) = 1;

            op.apply(x, Ax);

            const auto weight = op.weight(0.707106769084930);
            CAPTURE(weight);

            THEN("Each detector value is equal to 4 * the weight of distance 0.7071")
            {
                CHECK_EQ(Ax[0], Approx(4 * weight).epsilon(0.01));
            }
        }

        WHEN("Setting only the voxels directly above on the ray to 1")
        {
            // set all voxels on the ray to 1
            x(1, 0) = 1;
            x(2, 1) = 1;
            x(3, 2) = 1;
            x(4, 3) = 1;

            op.apply(x, Ax);

            const auto weight = op.weight(0.707106769084930);
            CAPTURE(weight);

            THEN("Each detector value is equal to 5 * the weight of distance 0.7071")
            {
                CHECK_EQ(Ax[0], Approx(4 * weight).epsilon(0.01));
            }
        }

        WHEN("Setting only the voxels two above on the ray")
        {
            // set all voxels on the ray to 1
            x(0, 2) = 1;
            x(1, 3) = 1;
            x(2, 4) = 1;

            op.apply(x, Ax);

            const auto distance = 2 * 0.707106769084930;
            const auto weight = op.weight(distance);
            CAPTURE(distance);
            CAPTURE(weight);

            THEN("Each detector value is equal to 3 * the weight of distance 0.7071")
            {
                CHECK_EQ(Ax[0], Approx(3 * weight));
            }
        }

        WHEN("Setting only the voxels directly above on the ray to 1")
        {
            // set all voxels on the ray to 1
            x(2, 0) = 1;
            x(3, 1) = 1;
            x(4, 2) = 1;

            op.apply(x, Ax);

            const auto distance = 2 * 0.707106769084930;
            const auto weight = op.weight(distance);
            CAPTURE(distance);
            CAPTURE(weight);

            THEN("Each detector value is equal to 3 * the weight of distance 0.7071")
            {
                CHECK_EQ(Ax[0], Approx(3 * weight));
            }
        }
    }

    GIVEN("a single detector of size 1, at 90 degree")
    {
        std::vector<Geometry> geom;
        geom.emplace_back(stc, ctr, Degree{90}, VolumeData2D{Size2D{sizeDomain}},
                          SinogramData2D{Size2D{sizeRange}});

        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BlobProjector(domain, range);

        auto Ax = DataContainer(range);
        Ax = 0;

        WHEN("Applying the BlobProjector to the volume")
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

            const auto weight = op.weight(1);
            CAPTURE(weight);

            THEN("Each detector value is equal to 5 * the weight of distance 1")
            {
                CHECK_EQ(Ax[0], Approx(5 * weight));
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

            const auto weight = op.weight(1);
            CAPTURE(weight);

            THEN("Each detector value is equal to 5 * the weight of distance 1")
            {
                CHECK_EQ(Ax[0], Approx(5 * weight));
            }
        }

        WHEN("Setting only the voxels 2 away neighbours above the ray to 1")
        {
            // set all voxels above the ray to 1, i.e the visited neighbours
            x(0, 4) = 1;
            x(1, 4) = 1;
            x(2, 4) = 1;
            x(3, 4) = 1;
            x(4, 4) = 1;

            op.apply(x, Ax);

            const auto weight = op.weight(2);
            CAPTURE(weight);

            THEN("Each detector value is equal to 5 * the weight of distance 2")
            {
                CHECK_EQ(Ax[0], Approx(5 * weight));
            }
        }

        WHEN("Setting only the voxels 2 away neighbours below the ray to 1")
        {
            // set all voxels above the ray to 1, i.e the visited neighbours
            x(0, 0) = 1;
            x(1, 0) = 1;
            x(2, 0) = 1;
            x(3, 0) = 1;
            x(4, 0) = 1;

            op.apply(x, Ax);

            const auto weight = op.weight(2);
            CAPTURE(weight);

            THEN("Each detector value is equal to 5 * the weight of distance 2")
            {
                CHECK_EQ(Ax[0], Approx(5 * weight));
            }
        }
    }

    GIVEN("a single detector of size 1, at 135 degree")
    {
        std::vector<Geometry> geom;
        geom.emplace_back(stc, ctr, Degree{135}, VolumeData2D{Size2D{sizeDomain}},
                          SinogramData2D{Size2D{sizeRange}});

        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BlobProjector(domain, range);

        auto Ax = DataContainer(range);
        Ax = 0;

        WHEN("Applying the BlobProjector to the volume")
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

            const auto distance = 0.707106769084930;
            const auto weight = op.weight(distance);
            CAPTURE(distance);
            CAPTURE(weight);

            THEN("Each detector value is equal to 4 * the weight of distance 0.7071")
            {
                CHECK_EQ(Ax[0], Approx(4 * weight).epsilon(0.01));
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

            const auto distance = 0.707106769084930;
            const auto weight = op.weight(distance);
            CAPTURE(distance);
            CAPTURE(weight);

            THEN("Each detector value is equal to 5 * the weight of distance 0.7071")
            {
                CHECK_EQ(Ax[0], Approx(4 * weight).epsilon(0.01));
            }
        }

        WHEN("Setting only the voxels two above on the ray")
        {
            // set all voxels on the ray to 1
            x(2, 4) = 1;
            x(3, 3) = 1;
            x(4, 2) = 1;

            op.apply(x, Ax);

            const auto distance = 2 * 0.707106769084930;
            const auto weight = op.weight(distance);
            CAPTURE(distance);
            CAPTURE(weight);

            THEN("Each detector value is equal to 3 * the weight of distance 0.7071")
            {
                CHECK_EQ(Ax[0], Approx(3 * weight));
            }
        }

        WHEN("Setting only the voxels directly above on the ray to 1")
        {
            // set all voxels on the ray to 1
            x(0, 2) = 1;
            x(1, 1) = 1;
            x(2, 0) = 1;

            op.apply(x, Ax);

            const auto distance = 2 * 0.707106769084930;
            const auto weight = op.weight(distance);
            CAPTURE(distance);
            CAPTURE(weight);

            THEN("Each detector value is equal to 3 * the weight of distance 0.7071")
            {
                CHECK_EQ(Ax[0], Approx(3 * weight));
            }
        }
    }

    GIVEN("a single detector of size 1, at 180 degree")
    {
        std::vector<Geometry> geom;
        geom.emplace_back(stc, ctr, Degree{180}, VolumeData2D{Size2D{sizeDomain}},
                          SinogramData2D{Size2D{sizeRange}});

        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BlobProjector(domain, range);

        auto Ax = DataContainer(range);
        Ax = 0;

        WHEN("Applying the BlobProjector to the volume")
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

            const auto weight = op.weight(1);
            CAPTURE(weight);

            THEN("Each detector value is equal to 5 * the weight of distance 1")
            {
                CHECK_EQ(Ax[0], Approx(5 * weight));
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

            const auto weight = op.weight(1);
            CAPTURE(weight);

            THEN("Each detector value is equal to 5 * the weight of distance 1")
            {
                CHECK_EQ(Ax[0], Approx(5 * weight));
            }
        }

        WHEN("Setting only the voxels 2 away neighbours above the ray to 1")
        {
            // set all voxels above the ray to 1, i.e the visited neighbours
            x(4, 0) = 1;
            x(4, 1) = 1;
            x(4, 2) = 1;
            x(4, 3) = 1;
            x(4, 4) = 1;

            op.apply(x, Ax);

            const auto weight = op.weight(2);
            CAPTURE(weight);

            THEN("Each detector value is equal to 5 * the weight of distance 2")
            {
                CHECK_EQ(Ax[0], Approx(5 * weight));
            }
        }

        WHEN("Setting only the voxels 2 away neighbours below the ray to 1")
        {
            // set all voxels above the ray to 1, i.e the visited neighbours
            x(0, 0) = 1;
            x(0, 1) = 1;
            x(0, 2) = 1;
            x(0, 3) = 1;
            x(0, 4) = 1;

            op.apply(x, Ax);

            const auto weight = op.weight(2);
            CAPTURE(weight);

            THEN("Each detector value is equal to 5 * the weight of distance 2")
            {
                CHECK_EQ(Ax[0], Approx(5 * weight));
            }
        }
    }

    GIVEN("a single detector of size 1, at 225 degree")
    {
        std::vector<Geometry> geom;
        geom.emplace_back(stc, ctr, Degree{225}, VolumeData2D{Size2D{sizeDomain}},
                          SinogramData2D{Size2D{sizeRange}});

        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BlobProjector(domain, range);

        auto Ax = DataContainer(range);
        Ax = 0;

        WHEN("Applying the BlobProjector to the volume")
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
        auto op = BlobProjector(domain, range);

        auto Ax = DataContainer(range);
        Ax = 0;

        WHEN("Applying the BlobProjector to the volume")
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
        auto op = BlobProjector(domain, range);

        auto Ax = DataContainer(range);
        Ax = 0;

        WHEN("Applying the BlobProjector to the volume")
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

TEST_CASE("LutProjector: Test single rays backward projection")
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
        auto op = BlobProjector(domain, range);

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

            THEN("The main direction of the ray is equal to the weight")
            {
                const auto weight = op.weight(1);
                CAPTURE(weight);

                CHECK_EQ(x(3, 0), Approx(weight));
                CHECK_EQ(x(3, 1), Approx(weight));
                CHECK_EQ(x(3, 2), Approx(weight));
                CHECK_EQ(x(3, 3), Approx(weight));
                CHECK_EQ(x(3, 4), Approx(weight));

                CHECK_EQ(x(1, 0), Approx(weight));
                CHECK_EQ(x(1, 1), Approx(weight));
                CHECK_EQ(x(1, 2), Approx(weight));
                CHECK_EQ(x(1, 3), Approx(weight));
                CHECK_EQ(x(1, 4), Approx(weight));
            }

            THEN("The main direction of the ray is equal to the weight")
            {
                const auto weight = op.weight(2);
                CAPTURE(weight);

                CHECK_EQ(x(4, 0), Approx(weight));
                CHECK_EQ(x(4, 1), Approx(weight));
                CHECK_EQ(x(4, 2), Approx(weight));
                CHECK_EQ(x(4, 3), Approx(weight));
                CHECK_EQ(x(4, 4), Approx(weight));

                CHECK_EQ(x(0, 0), Approx(weight));
                CHECK_EQ(x(0, 1), Approx(weight));
                CHECK_EQ(x(0, 2), Approx(weight));
                CHECK_EQ(x(0, 3), Approx(weight));
                CHECK_EQ(x(0, 4), Approx(weight));
            }
        }
    }

    GIVEN("a single detector of size 1, at 180 degree")
    {
        std::vector<Geometry> geom;
        geom.emplace_back(stc, ctr, Degree{180}, VolumeData2D{Size2D{sizeDomain}},
                          SinogramData2D{Size2D{sizeRange}});

        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BlobProjector(domain, range);

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

            THEN("The main direction of the ray is equal to the weight")
            {
                const auto weight = op.weight(1);
                CAPTURE(weight);

                CHECK_EQ(x(3, 0), Approx(weight));
                CHECK_EQ(x(3, 1), Approx(weight));
                CHECK_EQ(x(3, 2), Approx(weight));
                CHECK_EQ(x(3, 3), Approx(weight));
                CHECK_EQ(x(3, 4), Approx(weight));

                CHECK_EQ(x(1, 0), Approx(weight));
                CHECK_EQ(x(1, 1), Approx(weight));
                CHECK_EQ(x(1, 2), Approx(weight));
                CHECK_EQ(x(1, 3), Approx(weight));
                CHECK_EQ(x(1, 4), Approx(weight));
            }

            THEN("The main direction of the ray is equal to the weight")
            {
                const auto weight = op.weight(2);
                CAPTURE(weight);

                CHECK_EQ(x(4, 0), Approx(weight));
                CHECK_EQ(x(4, 1), Approx(weight));
                CHECK_EQ(x(4, 2), Approx(weight));
                CHECK_EQ(x(4, 3), Approx(weight));
                CHECK_EQ(x(4, 4), Approx(weight));

                CHECK_EQ(x(0, 0), Approx(weight));
                CHECK_EQ(x(0, 1), Approx(weight));
                CHECK_EQ(x(0, 2), Approx(weight));
                CHECK_EQ(x(0, 3), Approx(weight));
                CHECK_EQ(x(0, 4), Approx(weight));
            }
        }
    }

    GIVEN("a single detector of size 1, at 90 degree")
    {
        std::vector<Geometry> geom;
        geom.emplace_back(stc, ctr, Degree{90}, VolumeData2D{Size2D{sizeDomain}},
                          SinogramData2D{Size2D{sizeRange}});

        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BlobProjector(domain, range);

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

            THEN("The main direction of the ray is equal to the weight")
            {
                const auto weight = op.weight(1);
                CAPTURE(weight);

                CHECK_EQ(x(0, 3), Approx(weight));
                CHECK_EQ(x(1, 3), Approx(weight));
                CHECK_EQ(x(2, 3), Approx(weight));
                CHECK_EQ(x(3, 3), Approx(weight));
                CHECK_EQ(x(4, 3), Approx(weight));

                CHECK_EQ(x(0, 1), Approx(weight));
                CHECK_EQ(x(1, 1), Approx(weight));
                CHECK_EQ(x(2, 1), Approx(weight));
                CHECK_EQ(x(3, 1), Approx(weight));
                CHECK_EQ(x(4, 1), Approx(weight));
            }

            THEN("The main direction of the ray is equal to the weight")
            {
                const auto weight = op.weight(2);
                CAPTURE(weight);

                CHECK_EQ(x(0, 4), Approx(weight));
                CHECK_EQ(x(1, 4), Approx(weight));
                CHECK_EQ(x(2, 4), Approx(weight));
                CHECK_EQ(x(3, 4), Approx(weight));
                CHECK_EQ(x(4, 4), Approx(weight));

                CHECK_EQ(x(0, 0), Approx(weight));
                CHECK_EQ(x(1, 0), Approx(weight));
                CHECK_EQ(x(2, 0), Approx(weight));
                CHECK_EQ(x(3, 0), Approx(weight));
                CHECK_EQ(x(4, 0), Approx(weight));
            }
        }
    }

    GIVEN("a single detector of size 1, at 270 degree")
    {
        std::vector<Geometry> geom;
        geom.emplace_back(stc, ctr, Degree{270}, VolumeData2D{Size2D{sizeDomain}},
                          SinogramData2D{Size2D{sizeRange}});

        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BlobProjector(domain, range);

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

            THEN("The main direction of the ray is equal to the weight")
            {
                const auto weight = op.weight(1);
                CAPTURE(weight);

                CHECK_EQ(x(0, 3), Approx(weight));
                CHECK_EQ(x(1, 3), Approx(weight));
                CHECK_EQ(x(2, 3), Approx(weight));
                CHECK_EQ(x(3, 3), Approx(weight));
                CHECK_EQ(x(4, 3), Approx(weight));

                CHECK_EQ(x(0, 1), Approx(weight));
                CHECK_EQ(x(1, 1), Approx(weight));
                CHECK_EQ(x(2, 1), Approx(weight));
                CHECK_EQ(x(3, 1), Approx(weight));
                CHECK_EQ(x(4, 1), Approx(weight));
            }

            THEN("The main direction of the ray is equal to the weight")
            {
                const auto weight = op.weight(2);
                CAPTURE(weight);

                CHECK_EQ(x(0, 4), Approx(weight));
                CHECK_EQ(x(1, 4), Approx(weight));
                CHECK_EQ(x(2, 4), Approx(weight));
                CHECK_EQ(x(3, 4), Approx(weight));
                CHECK_EQ(x(4, 4), Approx(weight));

                CHECK_EQ(x(0, 0), Approx(weight));
                CHECK_EQ(x(1, 0), Approx(weight));
                CHECK_EQ(x(2, 0), Approx(weight));
                CHECK_EQ(x(3, 0), Approx(weight));
                CHECK_EQ(x(4, 0), Approx(weight));
            }
        }
    }

    GIVEN("a single detector of size 1, at 45 degree")
    {
        std::vector<Geometry> geom;
        geom.emplace_back(stc, ctr, Degree{45}, VolumeData2D{Size2D{sizeDomain}},
                          SinogramData2D{Size2D{sizeRange}});

        auto range = PlanarDetectorDescriptor(sizeRange, geom);
        auto op = BlobProjector(domain, range);

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

            THEN("The main direction of the ray is equal to the weight")
            {
                const auto distance = 0.707106769084930;
                const auto weight = op.weight(distance);
                CAPTURE(distance);
                CAPTURE(weight);

                CHECK_EQ(x(0, 1), Approx(weight));
                CHECK_EQ(x(1, 2), Approx(weight));
                CHECK_EQ(x(2, 3), Approx(weight));
                CHECK_EQ(x(3, 4), Approx(weight));

                CHECK_EQ(x(1, 0), Approx(weight));
                CHECK_EQ(x(2, 1), Approx(weight));
                CHECK_EQ(x(3, 2), Approx(weight));
                CHECK_EQ(x(4, 3), Approx(weight));
            }

            THEN("The main direction of the ray is equal to the weight")
            {
                const auto distance = 2 * 0.707106769084930;
                const auto weight = op.weight(distance);
                CAPTURE(weight);

                CHECK_EQ(x(0, 2), Approx(weight));
                CHECK_EQ(x(1, 3), Approx(weight));
                CHECK_EQ(x(2, 4), Approx(weight));

                CHECK_EQ(x(2, 0), Approx(weight));
                CHECK_EQ(x(3, 1), Approx(weight));
                CHECK_EQ(x(4, 2), Approx(weight));
            }
        }
    }
}

/*
TEST_CASE_TEMPLATE("BlobProjector: Test weights", data_t, float, double)
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
    auto op = BlobProjector<data_t, 101>(domain, range);

    for (int i = 0; i < 50; ++i) {
        const auto distance = i / 25.;

        CAPTURE(i);
        CAPTURE(distance);
        CHECK_EQ(Approx(op.weight(distance)), expected[i]);
    }
}*/

TEST_SUITE_END();
