/**
 * \file test_Intersection.cpp
 *
 * \brief Test for Intersection class
 *
 * \author David Frank - initial code
 * \author Maximilian Hornung - modularization
 * \author Tobias Lasser - consistency changes
 */

#include <catch2/catch.hpp>

#include "Intersection.h"

using namespace elsa;

using Ray = Eigen::ParametrizedLine<real_t, Eigen::Dynamic>;

SCENARIO("Intersect corners of pixels")
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
    Ray r(ro, rd);

    REQUIRE(Intersection::withRay(aabb, r));

    // top right corner
    ro << 1, 2;
    rd << 1.0, -1.0;
    rd.normalize();
    r = Ray(ro, rd);

    REQUIRE(!Intersection::withRay(aabb, r));

    // bottom left corner
    ro << 3, -2;
    rd << -1.0, 1.0;
    rd.normalize();
    r = Ray(ro, rd);

    REQUIRE(Intersection::withRay(aabb, r));

    // bottom right corner
    ro << 3, 1;
    rd << -1.0, -1.0;
    rd.normalize();
    r = Ray(ro, rd);

    REQUIRE(!Intersection::withRay(aabb, r));
}

SCENARIO("Intersect edges of voxels")
{
    GIVEN("some ray")
    {
        size_t dim = 2;
        RealVector_t ro(dim);
        ro << 132, 30;
        RealVector_t rd(dim);
        rd << -1.0, 0;
        Ray r(ro, rd);

        IndexVector_t voxel(dim);

        // horizontal check
        voxel << 30, 31;
        THEN("the ray intersects")
        {
            BoundingBox aabb(voxel);
            REQUIRE(Intersection::withRay(aabb, r));
        }

        voxel << 40, 30;
        THEN("the ray intersects")
        {
            BoundingBox aabb(voxel);
            REQUIRE(Intersection::withRay(aabb, r));
        }

        voxel << 40, 29;
        THEN("the ray does not intersect")
        {
            BoundingBox aabb(voxel);
            REQUIRE(!Intersection::withRay(aabb, r));
        }

        // vertical check
        ro << 30, -35;
        rd << 0, 1.0;
        r = Ray(ro, rd);

        voxel << 31, 30;
        THEN("the ray intersects")
        {
            BoundingBox aabb(voxel);
            REQUIRE(Intersection::withRay(aabb, r));
        }

        voxel << 30, 40;
        THEN("the ray intersects")
        {
            BoundingBox aabb(voxel);
            REQUIRE(Intersection::withRay(aabb, r));
        }

        voxel << 29, 40;
        THEN("the ray does not intersect")
        {
            BoundingBox aabb(voxel);
            REQUIRE(!Intersection::withRay(aabb, r));
        }

        rd << 0.0, 1.0;
        ro << 1.5, -10;
        r = Ray(ro, rd);

        voxel << 0, 1;
        THEN("the ray does not intersect")
        {
            BoundingBox aabb(voxel);
            REQUIRE(!Intersection::withRay(aabb, r));
        }

        voxel << 1, 1;
        THEN("the ray does not intersect")
        {
            BoundingBox aabb(voxel);
            REQUIRE(!Intersection::withRay(aabb, r));
        }

        voxel << 2, 1;
        THEN("the ray intersects")
        {
            BoundingBox aabb(voxel);
            REQUIRE(Intersection::withRay(aabb, r));
        }
    }
}
