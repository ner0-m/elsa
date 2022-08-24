/**
 * @file test_Intersection.cpp
 *
 * @brief Test for Intersection class
 *
 * @author David Frank - initial code
 * @author Maximilian Hornung - modularization
 * @author Tobias Lasser - consistency changes
 */

#include "doctest/doctest.h"

#include "Intersection.h"

using namespace elsa;
using namespace doctest;

#include "PrettyPrint/Stl.h"
#include "PrettyPrint/Eigen.h"

TEST_CASE("Intersection: Intersect corners of pixels")
{
    size_t dim = 2;

    IndexVector_t voxel(dim);
    voxel << 1, 0;
    BoundingBox aabb(voxel);

    // top left corner
    RealVector_t ro(dim);
    ro << -3, -3;
    RealVector_t rd(dim);
    rd << 1.0, 1.0;
    rd.normalize();
    RealRay_t r(ro, rd);

    REQUIRE_UNARY(Intersection::withRay(aabb, r));

    // top right corner
    ro << 1, 2;
    rd << 1.0, -1.0;
    rd.normalize();
    r = RealRay_t(ro, rd);

    REQUIRE_UNARY_FALSE(Intersection::withRay(aabb, r));

    // bottom left corner
    ro << 3, -2;
    rd << -1.0, 1.0;
    rd.normalize();
    r = RealRay_t(ro, rd);

    REQUIRE_UNARY(Intersection::withRay(aabb, r));

    // bottom right corner
    ro << 3, 1;
    rd << -1.0, -1.0;
    rd.normalize();
    r = RealRay_t(ro, rd);

    REQUIRE_UNARY_FALSE(Intersection::withRay(aabb, r));
}

TEST_CASE("Intersection: Intersect edges of voxels")
{
    GIVEN("A ray which intersects the edge of a voxel")
    {
        size_t dim = 2;
        RealVector_t ro(dim);
        ro << 132, 30;
        RealVector_t rd(dim);
        rd << -1.0, 0;
        RealRay_t r(ro, rd);

        IndexVector_t voxel(dim);

        // horizontal check
        voxel << 30, 31;
        THEN("the ray intersects")
        {
            BoundingBox aabb(voxel);
            REQUIRE_UNARY(Intersection::withRay(aabb, r));
        }

        voxel << 40, 30;
        THEN("the ray intersects")
        {
            BoundingBox aabb(voxel);
            REQUIRE_UNARY(Intersection::withRay(aabb, r));
        }

        voxel << 40, 29;
        THEN("the ray does not intersect")
        {
            BoundingBox aabb(voxel);
            REQUIRE_UNARY_FALSE(Intersection::withRay(aabb, r));
        }

        // vertical check
        ro << 30, -35;
        rd << 0, 1.0;
        r = RealRay_t(ro, rd);

        voxel << 31, 30;
        THEN("the ray intersects")
        {
            BoundingBox aabb(voxel);
            REQUIRE_UNARY(Intersection::withRay(aabb, r));
        }

        voxel << 30, 40;
        THEN("the ray intersects")
        {
            BoundingBox aabb(voxel);
            REQUIRE_UNARY(Intersection::withRay(aabb, r));
        }

        voxel << 29, 40;
        THEN("the ray does not intersect")
        {
            BoundingBox aabb(voxel);
            REQUIRE_UNARY_FALSE(Intersection::withRay(aabb, r));
        }

        rd << 0.0, 1.0;
        ro << 1.5, -10;
        r = RealRay_t(ro, rd);

        voxel << 0, 1;
        THEN("the ray does not intersect")
        {
            BoundingBox aabb(voxel);
            REQUIRE_UNARY_FALSE(Intersection::withRay(aabb, r));
        }

        voxel << 1, 1;
        THEN("the ray does not intersect")
        {
            BoundingBox aabb(voxel);
            REQUIRE_UNARY_FALSE(Intersection::withRay(aabb, r));
        }

        voxel << 2, 1;
        THEN("the ray intersects")
        {
            BoundingBox aabb(voxel);
            REQUIRE_UNARY(Intersection::withRay(aabb, r));
        }
    }
}

TEST_SUITE_BEGIN("Intersection::voxelCenter");

Eigen::IOFormat vecfmt(8, 0, ", ", ", ", "", "", "[", "]");

TEST_CASE("Intersection: Ray with voxel center")
{
    IndexVector_t upperbound({{5, 5}});
    BoundingBox aabb(upperbound);

    GIVEN("Ray entering the bottom of the volume")
    {
        // Check multiple different rays all coming from the bottom
        auto steps = 30;
        for (int i = 0; i <= steps; ++i) {
            auto inc = (1.f / static_cast<float>(steps)) * static_cast<float>(i);
            auto dirinc = static_cast<float>(i) / (2 * static_cast<float>(steps));

            RealVector_t origin({{0.75f + inc, 0}});
            RealVector_t dir({{1, 0.5f + dirinc}});
            dir.normalize();

            CAPTURE(inc);
            CAPTURE(origin.format(vecfmt));
            CAPTURE(dir.format(vecfmt));

            RealRay_t ray(origin, dir);

            auto hit = Intersection::xPlanesWithRay(aabb, ray);

            CAPTURE(hit);
            REQUIRE(hit);

            THEN("entries x-coord lies in the middle of a voxel")
            {
                RealVector_t enter = ray.pointAt(hit->_tmin);
                CAPTURE(enter.format(vecfmt));

                real_t trash = 0;
                CHECK_EQ(0.5, doctest::Approx(std::modf(enter[0], &trash)));

                CHECK((enter.array() < upperbound.template cast<real_t>().array()).all());
                CHECK((enter.array() > 0).all());
            }

            THEN("x-coord lies in the middle of a voxel")
            {
                RealVector_t leave = ray.pointAt(hit->_tmax);
                CAPTURE(leave.format(vecfmt));

                real_t trash = 0;
                CHECK_EQ(0.5, doctest::Approx(std::modf(leave[0], &trash)));

                CHECK((leave.array() < upperbound.template cast<real_t>().array()).all());
                CHECK((leave.array() > 0).all());
            }
        }
    }

    GIVEN("Ray entering the top of the volume")
    {
        // Check multiple different rays all coming from the top
        auto steps = 30;
        for (int i = 0; i <= steps; ++i) {
            auto inc = (1.f / static_cast<float>(steps)) * static_cast<float>(i);
            auto dirinc = static_cast<float>(i) / (2 * static_cast<float>(steps));

            RealVector_t origin({{0.75f + inc, 6}});
            RealVector_t dir({{1, -0.5f - dirinc}});
            dir.normalize();

            CAPTURE(inc);
            CAPTURE(origin.format(vecfmt));
            CAPTURE(dir.format(vecfmt));

            RealRay_t ray(origin, dir);

            auto hit = Intersection::xPlanesWithRay(aabb, ray);

            CAPTURE(hit);
            REQUIRE(hit);

            THEN("entries x-coord lies in the middle of a voxel")
            {
                RealVector_t enter = ray.pointAt(hit->_tmin);
                CAPTURE(enter.format(vecfmt));

                real_t trash = 0;
                CHECK_EQ(0.5, doctest::Approx(std::modf(enter[0], &trash)));

                CHECK((enter.array() < upperbound.template cast<real_t>().array()).all());
                CHECK((enter.array() > 0).all());
            }

            THEN("x-coord lies in the middle of a voxel")
            {
                RealVector_t leave = ray.pointAt(hit->_tmax);
                CAPTURE(leave.format(vecfmt));

                real_t trash = 0;
                CHECK_EQ(0.5, doctest::Approx(std::modf(leave[0], &trash)));

                CHECK((leave.array() < upperbound.template cast<real_t>().array()).all());
                CHECK((leave.array() > 0).all());
            }
        }
    }
}

TEST_CASE("Intersection: Quick bug tests")
{
    const IndexVector_t size({{3, 3}});

    BoundingBox aabb(size);
    aabb.min() = RealVector_t({{-1.5f, -1.5f}});
    aabb.max() = RealVector_t({{1.5f, 1.5f}});

    const RealVector_t ro({{-2, -3}});
    RealVector_t rd({{1, 1}});
    rd.normalize();
    const RealRay_t ray(ro, rd);

    auto hit = Intersection::xPlanesWithRay(aabb, ray);

    CAPTURE(hit);
    REQUIRE(hit);

    auto dist_to_integer = [&](auto f) {
        real_t aabbmin = static_cast<int>(std::round(f));
        auto frac = std::abs(f - aabbmin);
        return frac;
    };

    THEN("entries x-coord lies in the middle of a voxel")
    {
        RealVector_t entry = ray.pointAt(hit->_tmin);
        CAPTURE(entry.format(vecfmt));

        // real_t trash = 0;
        CHECK_EQ(0, doctest::Approx(dist_to_integer(entry[0])));

        CHECK((entry.array() < 1.5).all());
        CHECK((entry.array() > -1.5).all());
    }

    THEN("x-coord lies in the middle of a voxel")
    {
        RealVector_t leave = ray.pointAt(hit->_tmax);
        CAPTURE(leave.format(vecfmt));

        // real_t trash = 0;
        // CHECK_EQ(0, doctest::Approx(std::modf(leave[0], &trash)));
        CHECK_EQ(0, doctest::Approx(dist_to_integer(leave[0])));

        CHECK((leave.array() < 1.5).all());
        CHECK((leave.array() > -1.5).all());
    }
}

TEST_CASE("Intersection: Quick bug tests")
{
    const IndexVector_t size({{3, 3}});

    BoundingBox aabb(size);
    aabb.min() = RealVector_t({{-1.5f, -1.5f}});
    aabb.max() = RealVector_t({{1.5f, 1.5f}});

    const RealVector_t ro({{-2, -3}});
    RealVector_t rd({{1, 1}});
    rd.normalize();
    const RealRay_t ray(ro, rd);

    auto hit = Intersection::xPlanesWithRay(aabb, ray);

    CAPTURE(hit);
    REQUIRE(hit);

    auto dist_to_integer = [&](auto f) {
        real_t aabbmin = static_cast<int>(std::round(f));
        auto frac = std::abs(f - aabbmin);
        return frac;
    };

    THEN("entries x-coord lies in the middle of a voxel")
    {
        RealVector_t entry = ray.pointAt(hit->_tmin);
        CAPTURE(entry.format(vecfmt));

        // real_t trash = 0;
        CHECK_EQ(0, doctest::Approx(dist_to_integer(entry[0])));

        CHECK((entry.array() < 1.5).all());
        CHECK((entry.array() > -1.5).all());
    }

    THEN("x-coord lies in the middle of a voxel")
    {
        RealVector_t leave = ray.pointAt(hit->_tmax);
        CAPTURE(leave.format(vecfmt));

        // real_t trash = 0;
        // CHECK_EQ(0, doctest::Approx(std::modf(leave[0], &trash)));
        CHECK_EQ(0, doctest::Approx(dist_to_integer(leave[0])));

        CHECK((leave.array() < 1.5).all());
        CHECK((leave.array() > -1.5).all());
    }
}

// Redefine GIVEN such that it's nicely usable inside an loop
#undef GIVEN
#define GIVEN(...) DOCTEST_SUBCASE((std::string("   Given: ") + std::string(__VA_ARGS__)).c_str())

TEST_CASE("Intersection: 3D rays with bounding box")
{
    const IndexVector_t upperbound({{5, 5, 5}});
    const BoundingBox aabb(upperbound);

    for (int z = 0; z < 25; ++z) {
        for (int y = 0; y < 25; ++y) {
            const auto zinc = static_cast<real_t>(z) / 5.f;
            const auto yinc = static_cast<real_t>(y) / 5.f;
            GIVEN("ray starting at (6, " + std::to_string(yinc) + ", " + std::to_string(zinc) + ")")
            {
                const RealVector_t origin({{-1, yinc, zinc}});
                RealVector_t dir({{1, 0, 0}});

                CAPTURE(yinc);
                CAPTURE(zinc);
                CAPTURE(origin.format(vecfmt));
                CAPTURE(dir.format(vecfmt));

                RealRay_t ray(origin, dir);

                auto hit = Intersection::xPlanesWithRay(aabb, ray);

                CAPTURE(hit);
                REQUIRE(hit);

                THEN("entries x-coord lies in the middle of a voxel")
                {
                    RealVector_t enter = ray.pointAt(hit->_tmin);
                    CAPTURE(enter.format(vecfmt));

                    real_t trash = 0;
                    CHECK_EQ(0.5, doctest::Approx(std::modf(enter[0], &trash)));

                    CHECK((enter.array() <= upperbound.template cast<real_t>().array()).all());
                    CHECK((enter.array() >= 0).all());
                }

                THEN("x-coord lies in the middle of a voxel")
                {
                    RealVector_t leave = ray.pointAt(hit->_tmax);
                    CAPTURE(leave.format(vecfmt));

                    real_t trash = 0;
                    CHECK_EQ(0.5, doctest::Approx(std::modf(leave[0], &trash)));

                    CHECK((leave.array() <= upperbound.template cast<real_t>().array()).all());
                    CHECK((leave.array() >= 0).all());
                }
            }
        }
    }

    for (int z = 0; z < 25; ++z) {
        for (int y = 0; y < 25; ++y) {
            const auto zinc = static_cast<real_t>(z) / 5.f;
            const auto yinc = static_cast<real_t>(y) / 5.f;
            GIVEN("ray starting at (6, " + std::to_string(yinc) + ", " + std::to_string(zinc) + ")")
            {
                const RealVector_t origin({{6, yinc, zinc}});
                RealVector_t dir({{-1, 0, 0}});

                CAPTURE(yinc);
                CAPTURE(zinc);
                CAPTURE(origin.format(vecfmt));
                CAPTURE(dir.format(vecfmt));

                RealRay_t ray(origin, dir);

                auto hit = Intersection::xPlanesWithRay(aabb, ray);

                CAPTURE(hit);
                REQUIRE(hit);

                THEN("entries x-coord lies in the middle of a voxel")
                {
                    RealVector_t enter = ray.pointAt(hit->_tmin);
                    CAPTURE(enter.format(vecfmt));

                    real_t trash = 0;
                    CHECK_EQ(0.5, doctest::Approx(std::modf(enter[0], &trash)));

                    CHECK((enter.array() <= upperbound.template cast<real_t>().array()).all());
                    CHECK((enter.array() >= 0).all());
                }

                THEN("x-coord lies in the middle of a voxel")
                {
                    RealVector_t leave = ray.pointAt(hit->_tmax);
                    CAPTURE(leave.format(vecfmt));

                    real_t trash = 0;
                    CHECK_EQ(0.5, doctest::Approx(std::modf(leave[0], &trash)));

                    CHECK((leave.array() <= upperbound.template cast<real_t>().array()).all());
                    CHECK((leave.array() >= 0).all());
                }
            }
        }
    }
}

TEST_CASE("Intersection: 3D rays with bounding box")
{
    const IndexVector_t upperbound({{5, 5, 5}});
    BoundingBox aabb(upperbound);
    aabb.min() = RealVector_t({{-2.5f, -2.5f, -2.5f}});
    aabb.max() = RealVector_t({{2.5f, 2.5f, 2.5f}});

    for (int z = 0; z < 25; ++z) {
        for (int y = 0; y < 25; ++y) {
            const auto zinc = static_cast<real_t>(z) / 5.f - 2.5f;
            const auto yinc = static_cast<real_t>(y) / 5.f - 2.5f;

            GIVEN("ray starting at (6, " + std::to_string(yinc) + ", " + std::to_string(zinc) + ")")
            {
                const RealVector_t origin({{-5, yinc, zinc}});
                RealVector_t dir({{1, 0, 0}});

                CAPTURE(yinc);
                CAPTURE(zinc);
                CAPTURE(origin.format(vecfmt));
                CAPTURE(dir.format(vecfmt));

                RealRay_t ray(origin, dir);

                auto hit = Intersection::xPlanesWithRay(aabb, ray);

                CAPTURE(hit);
                REQUIRE(hit);

                THEN("entries x-coord lies in the middle of a voxel")
                {
                    RealVector_t enter = ray.pointAt(hit->_tmin);
                    CAPTURE(enter.format(vecfmt));

                    CHECK_EQ(-2, doctest::Approx(enter[0]));

                    CHECK((enter.array() <= 2.5).all());
                    CHECK((enter.array() >= -2.5).all());
                }

                THEN("x-coord lies in the middle of a voxel")
                {
                    RealVector_t leave = ray.pointAt(hit->_tmax);
                    CAPTURE(leave.format(vecfmt));

                    CHECK_EQ(2, doctest::Approx(leave[0]));

                    CHECK((leave.array() <= 2.5).all());
                    CHECK((leave.array() >= -2.5).all());
                }
            }

            GIVEN("ray starting at (6, " + std::to_string(yinc) + ", " + std::to_string(zinc) + ")")
            {
                const RealVector_t origin({{6, yinc, zinc}});
                RealVector_t dir({{-1, 0, 0}});

                CAPTURE(yinc);
                CAPTURE(zinc);
                CAPTURE(origin.format(vecfmt));
                CAPTURE(dir.format(vecfmt));

                RealRay_t ray(origin, dir);

                auto hit = Intersection::xPlanesWithRay(aabb, ray);

                CAPTURE(hit);
                REQUIRE(hit);

                THEN("entries x-coord lies in the middle of a voxel")
                {
                    RealVector_t enter = ray.pointAt(hit->_tmin);
                    CAPTURE(enter.format(vecfmt));

                    CHECK_EQ(2, doctest::Approx(enter[0]));

                    CHECK((enter.array() <= 2.5).all());
                    CHECK((enter.array() >= -2.5).all());
                }

                THEN("x-coord lies in the middle of a voxel")
                {
                    RealVector_t leave = ray.pointAt(hit->_tmax);
                    CAPTURE(leave.format(vecfmt));

                    CHECK_EQ(-2, doctest::Approx(leave[0]));

                    CHECK((leave.array() <= 2.5).all());
                    CHECK((leave.array() >= -2.5).all());
                }
            }
        }
    }
}

TEST_SUITE_END();
