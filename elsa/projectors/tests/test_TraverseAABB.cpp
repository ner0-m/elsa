/**
 * \file test_TraverseAABB.cpp
 *
 * \brief Test for TraverseAABB class
 *
 * \author David Frank - initial code
 * \author Maximilian Hornung - modularization, fixes
 * \author Tobias Lasser - minor fixes
 */

#include <catch2/catch.hpp>

#include "TraverseAABB.h"
#include "Intersection.h"

using namespace elsa;

using Ray = Eigen::ParametrizedLine<real_t, Eigen::Dynamic>;


bool intersect(const RealVector_t& voxel, const Ray& r) {
    // pre-check parallel rays
    for (index_t i = 0; i < r.dim(); ++i) {
        real_t tmp = std::abs(r.origin()(i) - voxel(i));

        if (std::abs(r.direction()(i)) < 0.0000001 && tmp >= 0.0 && tmp < 1.0)
            return true;
    }

    // check if ray intersects pixel
    IndexVector_t ones(voxel.size());
    ones.setOnes();
    BoundingBox bb(ones);
    bb._min += voxel;
    bb._max += voxel;

    return Intersection::withRay(bb, r).operator bool();
}


SCENARIO("Construction of a 2D traversal object") {
    // setup
    size_t dim = 2;
    size_t x = 3;
    size_t y = 3;
    IndexVector_t volumeDims(dim);
    volumeDims << x, y;

    RealVector_t spacing(dim);

    RealVector_t ro(dim);
    RealVector_t rd(dim);

    GIVEN("A 3x3 aabb with standard spacing") {
        BoundingBox aabb(volumeDims);

        //================================
        //  intersection from straight rays from the bottom
        //================================
        WHEN("A traversal algorithms is initialised with the aabb and a ray with origin = (0.5, -0.5) and direction (0, 1)") {
            ro << 0.5, -0.5;
            rd << 0.0, 1.0;
            Ray r(ro, rd);

            TraverseAABB traverse(aabb, r);
            CHECK(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel") {
                REQUIRE(traverse.getCurrentVoxel()(0) == 0);
                REQUIRE(traverse.getCurrentVoxel()(1) == 0);
            }
        }

        WHEN("A ray with origin = (1.0, -0.5) and direction (0, 1), hits the boundary between 2 voxels") {
            ro << 1.0, -0.5;
            rd << 0.0, 1.0;
            Ray r(ro, rd);

            TraverseAABB traverse(aabb, r);
            CHECK(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel") {
                REQUIRE(traverse.getCurrentVoxel()(0) == 1);
                REQUIRE(traverse.getCurrentVoxel()(1) == 0);
            }
        }

        WHEN("A traversal algorithms is initialised with the aabb and a ray with origin = (1.5, -0.5) and direction (0, 1)") {
            ro << 1.5, -0.5;
            rd << 0.0, 1.0;
            Ray r(ro, rd);

            TraverseAABB traverse(aabb, r);
            CHECK(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel") {
                REQUIRE(traverse.getCurrentVoxel()(0) == 1);
                REQUIRE(traverse.getCurrentVoxel()(1) == 0);
            }
        }

        WHEN("A ray with origin = (2.0, -0.5) and direction (0, 1), hits the boundary between 2 voxels") {
            ro << 2.0, -0.5;
            rd << 0.0, 1.0;
            Ray r(ro, rd);

            TraverseAABB traverse(aabb, r);
            CHECK(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel") {
                REQUIRE(traverse.getCurrentVoxel()(0) == 2);
                REQUIRE(traverse.getCurrentVoxel()(1) == 0);
            }
        }

        WHEN("A traversal algorithms is initialised with the aabb and a ray with origin = (2.5, -0.5) and direction (0, 1)") {
            ro << 2.5, -0.5;
            rd << 0.0, 1.0;
            Ray r(ro, rd);

            TraverseAABB traverse(aabb, r);
            CHECK(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel") {
                REQUIRE(traverse.getCurrentVoxel()(0) == 2);
                REQUIRE(traverse.getCurrentVoxel()(1) == 0);
            }
        }


        //================================
        //  intersection from straight rays from the left
        //================================
        WHEN("A traversal algorithms is initialised with the aabb and a ray with origin = (-0.5, 0.5) and direction (1, 0)") {
            ro << -0.5, 0.5;
            rd << 1.0, 0.0;
            Ray r(ro, rd);

            TraverseAABB traverse(aabb, r);
            CHECK(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel") {
                REQUIRE(traverse.getCurrentVoxel()(0) == 0);
                REQUIRE(traverse.getCurrentVoxel()(1) == 0);
            }
        }

        WHEN("A ray with origin = (1.0, -0.5) and direction (0, 1), hits the boundary between 2 voxels") {
            ro << -0.5, 1.0;
            rd << 1.0, 0.0;
            Ray r(ro, rd);

            TraverseAABB traverse(aabb, r);
            CHECK(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel") {
                REQUIRE(traverse.getCurrentVoxel()(0) == 0);
                REQUIRE(traverse.getCurrentVoxel()(1) == 1);
            }
        }

        WHEN("A traversal algorithms is initialised with the aabb and a ray with origin = (-0.5, 1.5) and direction (1, 0)") {
            ro << -0.5, 1.5;
            rd << 1.0, 0.0;
            Ray r(ro, rd);

            TraverseAABB traverse(aabb, r);
            CHECK(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel") {
                REQUIRE(traverse.getCurrentVoxel()(0) == 0);
                REQUIRE(traverse.getCurrentVoxel()(1) == 1);
            }
        }

        WHEN("A ray with origin = (2.0, -0.5) and direction (0, 1), hits the boundary between 2 voxels") {
            ro << -0.5, 2.0;
            rd << 1.0, 0.0;
            Ray r(ro, rd);

            TraverseAABB traverse(aabb, r);
            CHECK(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel") {
                REQUIRE(traverse.getCurrentVoxel()(0) == 0);
                REQUIRE(traverse.getCurrentVoxel()(1) == 2);
            }
        }

        WHEN("A traversal algorithms is initialised with the aabb and a ray with origin = (-0.5, 2.5) and direction (1, 0)") {
            ro << -0.5, 2.5;
            rd << 1.0, 0.0;
            Ray r(ro, rd);

            TraverseAABB traverse(aabb, r);
            CHECK(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel") {
                REQUIRE(traverse.getCurrentVoxel()(0) == 0);
                REQUIRE(traverse.getCurrentVoxel()(1) == 2);
            }
        }

        //================================
        //  intersection from straight rays from the right
        //================================
        WHEN("A traversal algorithms is initialised with the aabb and a ray with origin = (3.5, 0.5) and direction (-1, 0)") {
            ro << 3.5, 0.5;
            rd << -1.0, 0.0;
            Ray r(ro, rd);

            TraverseAABB traverse(aabb, r);
            CHECK(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel") {
                REQUIRE(traverse.getCurrentVoxel()(0) == 2);
                REQUIRE(traverse.getCurrentVoxel()(1) == 0);
            }
        }

        WHEN("A ray with origin = (3.5, 1.0) and direction (-1, 0), hits the boundary between 2 voxels") {
            ro << 3.5, 1.0;
            rd << -1.0, 0.0;
            Ray r(ro, rd);

            TraverseAABB traverse(aabb, r);
            CHECK(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel") {
                REQUIRE(traverse.getCurrentVoxel()(0) == 2);
                REQUIRE(traverse.getCurrentVoxel()(1) == 1);
            }
        }

        WHEN("A traversal algorithms is initialised with the aabb and a ray with origin = (3.5, 1.5) and direction (-1, 0)") {
            ro << 3.5, 1.5;
            rd << -1.0, 0.0;
            Ray r(ro, rd);

            TraverseAABB traverse(aabb, r);
            CHECK(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel") {
                REQUIRE(traverse.getCurrentVoxel()(0) == 2);
                REQUIRE(traverse.getCurrentVoxel()(1) == 1);
            }
        }

        WHEN("A ray with origin = (3.5, 2.0) and direction (-1, 0), hits the boundary between 2 voxels") {
            ro << 3.5, 2.0;
            rd << -1.0, 0.0;
            Ray r(ro, rd);

            TraverseAABB traverse(aabb, r);
            CHECK(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel") {
                REQUIRE(traverse.getCurrentVoxel()(0) == 2);
                REQUIRE(traverse.getCurrentVoxel()(1) == 2);
            }
        }

        WHEN("A traversal algorithms is initialised with the aabb and a ray with origin = (3.5, 2.5) and direction (-1, 0)") {
            ro << 3.5, 2.5;
            rd << -1.0, 0.0;
            Ray r(ro, rd);

            TraverseAABB traverse(aabb, r);
            CHECK(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel") {
                REQUIRE(traverse.getCurrentVoxel()(0) == 2);
                REQUIRE(traverse.getCurrentVoxel()(1) == 2);
            }
        }

        //================================
        //  intersection from straight rays from the top
        //================================
        WHEN("A traversal algorithms is initialised with the aabb and a ray with origin = (0.5, 3.5) and direction (0, -1)") {
            ro << 0.5, 3.5;
            rd << 0.0, -1.0;
            Ray r(ro, rd);

            TraverseAABB traverse(aabb, r);
            CHECK(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel") {
                REQUIRE(traverse.getCurrentVoxel()(0) == 0);
                REQUIRE(traverse.getCurrentVoxel()(1) == 2);
            }
        }

        WHEN("A ray with origin = (1.0, 3.5) and direction (-1, 0), hits the boundary between 2 voxels") {
            ro << 1.0, 3.5;
            rd << 0.0, -1.0;
            Ray r(ro, rd);

            TraverseAABB traverse(aabb, r);
            CHECK(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel") {
                REQUIRE(traverse.getCurrentVoxel()(0) == 1);
                REQUIRE(traverse.getCurrentVoxel()(1) == 2);
            }
        }

        WHEN("A traversal algorithms is initialised with the aabb and a ray with origin = (1.5, 3.5) and direction (0, -1)") {
            ro << 1.5, 3.5;
            rd << 0.0, -1.0;
            Ray r(ro, rd);

            TraverseAABB traverse(aabb, r);
            CHECK(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel") {
                REQUIRE(traverse.getCurrentVoxel()(0) == 1);
                REQUIRE(traverse.getCurrentVoxel()(1) == 2);
            }
        }

        WHEN("A ray with origin = (2.0, 3.5) and direction (-1, 0), hits the boundary between 2 voxels") {
            ro << 2.0, 3.5;
            rd << 0.0, -1.0;
            Ray r(ro, rd);

            TraverseAABB traverse(aabb, r);
            CHECK(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel") {
                REQUIRE(traverse.getCurrentVoxel()(0) == 2);
                REQUIRE(traverse.getCurrentVoxel()(1) == 2);
            }
        }

        WHEN("A traversal algorithms is initialised with the aabb and a ray with origin = (2.5, 3.5) and direction (0, -1)") {
            ro << 2.5, 3.5;
            rd << 0.0, -1.0;
            Ray r(ro, rd);

            TraverseAABB traverse(aabb, r);
            CHECK(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel") {
                REQUIRE(traverse.getCurrentVoxel()(0) == 2);
                REQUIRE(traverse.getCurrentVoxel()(1) == 2);
            }
        }

        //
        // Some edge cases
        //
        WHEN("A ray with origin = (-0.5, 0.0) and direction (1, 0) hits the left edge of aabb") {
            ro << -0.5, 0.0;
            rd << 1.0, 0.0;
            Ray r(ro, rd);

            TraverseAABB traverse(aabb, r);

            THEN("the the aabb is not hit") {
                REQUIRE_FALSE(traverse.isInBoundingBox());
            }
        }

        WHEN("A ray with origin = (3.5, 0.0) and direction (-1, 0) hits the left edge of aabb") {
            ro << 3.5, 0.0;
            rd << -1.0, 0.0;
            Ray r(ro, rd);

            TraverseAABB traverse(aabb, r);

            THEN("the the aabb is not hit") {
                REQUIRE_FALSE(traverse.isInBoundingBox());
            }
        }

        WHEN("A ray with origin = (-0.5, 3.0) and direction (1, 0) hits the top edge of aabb") {
            ro << -0.5, 3.0;
            rd << 1.0, 0.0;
            Ray r(ro, rd);

            TraverseAABB traverse(aabb, r);

            THEN("the the aabb is not hit") {
                REQUIRE_FALSE(traverse.isInBoundingBox());
            }
        }

        WHEN("A ray with origin = (3.5, 3.0) and direction (-1, 0) hits the top edge of aabb") {
            ro << 3.5, 3.0;
            rd << -1.0, 0.0;
            Ray r(ro, rd);

            TraverseAABB traverse(aabb, r);

            THEN("the the aabb is not hit") {
                REQUIRE_FALSE(traverse.isInBoundingBox());
            }
        }

        WHEN("A ray with origin = (0.0, -0.5) and direction (0, 1) hits the bottom edge of aabb)") {
            ro << 0.0, -0.5;
            rd << 0.0, 1.0;
            Ray r(ro, rd);

            TraverseAABB traverse(aabb, r);

            THEN("the the aabb is not hit") {
                REQUIRE_FALSE(traverse.isInBoundingBox());
            }
        }

        WHEN("A ray with origin = (0.0, 3.5) and direction (0, -1) hits the top edge of aabb)") {
            ro << 0.0, 3.5;
            rd << 0.0, -1.0;
            Ray r(ro, rd);

            TraverseAABB traverse(aabb, r);

            THEN("the the aabb is not hit") {
                REQUIRE_FALSE(traverse.isInBoundingBox());
            }
        }

        WHEN("A ray with origin = (3.0, -0.5) and direction (0, 1) hits the right edge of aabb)") {
            ro << 3.0, -0.5;
            rd << 0.0, 1.0;
            Ray r(ro, rd);

            TraverseAABB traverse(aabb, r);

            THEN("the the aabb is not hit") {
                REQUIRE_FALSE(traverse.isInBoundingBox());
            }
        }

        WHEN("A ray with origin = (3.0, 3.5) and direction (0, -1) hits the top edge of aabb)") {
            ro << 3.0, 3.5;
            rd << 0.0, -1.0;
            Ray r(ro, rd);

            TraverseAABB traverse(aabb, r);

            THEN("the the aabb is not hit") {
                REQUIRE_FALSE(traverse.isInBoundingBox());
            }
        }
    }
}

SCENARIO("Construction of a 3D traversal object")
{
    // setup
    size_t dim = 3;
    IndexVector_t volumeDims(dim);
    volumeDims << 3, 3, 3;

    RealVector_t spacing(dim);

    RealVector_t ro(dim);
    RealVector_t rd(dim);

    GIVEN("a 3x3x3 aabb with standard spacing") {
        BoundingBox aabb(volumeDims);

        WHEN("A traversal algorithms is initialised with the aabb and a ray with origin = (0.5, -0.5, 0.5) and direction = (0, 1, 0)") {
            ro << 0.5, -0.5, 0.5;
            rd << 0.0, 1.0, 0.0;
            Ray r(ro, rd);

            TraverseAABB traverse(aabb, r);
            CHECK(traverse.isInBoundingBox());

            THEN("The ray intersects the aabb at the voxel (0, 0, 0)") {
                REQUIRE(traverse.isInBoundingBox());
                REQUIRE(traverse.getCurrentVoxel()(0) == 0);
                REQUIRE(traverse.getCurrentVoxel()(1) == 0);
                REQUIRE(traverse.getCurrentVoxel()(2) == 0);
            }
        }
    }


}



SCENARIO("Traverse a minimal 3D volume of size 1x1x1") {
    // setup
    size_t dim = 3;
    size_t x = 1;
    size_t y = 1;
    size_t z = 1;
    IndexVector_t volumeDims(dim);
    volumeDims << x, y, z;

    RealVector_t spacing(dim);
    RealVector_t ro(dim);
    RealVector_t rd(dim);

    GIVEN("A 1x1x1 volume with uniform scaling") {
        BoundingBox aabb(volumeDims);
        spacing << 1.0, 1.0, 1.0;

        WHEN("The volume is traversed with a ray with origin = (-0.5, 0.5, 0.5) and a direction = (0, 1, 0)") {
            ro << 0.5, -0.5, 0.5;
            rd << 0.0, 1.0, 0.0;

            Ray r(ro, rd);
            TraverseAABB traverse(aabb, r);
            CHECK(traverse.isInBoundingBox());

            traverse.updateTraverse();

            THEN("The algorithms left the volume and the voxel it left the box is (0, 1, 0)") {
                REQUIRE_FALSE(traverse.isInBoundingBox());
                REQUIRE(traverse.getCurrentVoxel()(0) == 0);
                REQUIRE(traverse.getCurrentVoxel()(1) == 1);
                REQUIRE(traverse.getCurrentVoxel()(2) == 0);
            }
        }
    }
}

SCENARIO("Traverse a 2D volume and only check that the endpoint is correct")
{
    // setup
    size_t dim = 2;
    size_t x = 10;
    size_t y = 10;
    IndexVector_t volumeDims(dim);
    volumeDims << x, y;

    RealVector_t spacing(dim);
    RealVector_t ro(dim);
    RealVector_t rd(dim);

    GIVEN("A 10x10 volume with uniform scaling")
    {
        BoundingBox aabb(volumeDims);

        WHEN("The volume is traversed with a ray with origin = (-0.5, 0.5, 0.5) and a direction = (0, 1, 0)")
        {
            ro << -1, 4.5;
            rd << 1.0, 0;

            Ray r(ro, rd);
            TraverseAABB traverse(aabb, r);
            CHECK(traverse.isInBoundingBox());

            while (traverse.isInBoundingBox())
                traverse.updateTraverse();

            THEN("The endpoint should be (10,4)") {
                REQUIRE_FALSE(traverse.isInBoundingBox());
                REQUIRE(traverse.getCurrentVoxel()(0) == 10);
                REQUIRE(traverse.getCurrentVoxel()(1) == 4);
            }
        }
    }
}

SCENARIO("Traverse a 3D Volume diagonally")
{
    // TODO: run through all 4 diagonals
    // TODO: make a non cube volume and run through all 4 diagonals
    // TODO: make non uniform scaling and run through all 4 diagonals

    size_t dim = 3;
    IndexVector_t volumeDims(dim);
    volumeDims << 10, 10, 10;

    RealVector_t spacing(dim);
    RealVector_t ro(dim);
    RealVector_t rd(dim);

    GIVEN("A 10x10 volume with uniform scaling")
    {
        BoundingBox aabb(volumeDims);
        WHEN("Start at (-1, -1, -1) (so bottom left front) and run to (10, 10, 10) (so top right back)")
        {
            ro << -1.0, -1.0, -1.0;
            rd << 1.0, 1.0, 1.0;
            rd.normalize();

            Ray r(ro, rd);

            TraverseAABB traverse(aabb, r);
            CHECK(traverse.isInBoundingBox());

            THEN("You entered at (0, 0, 0)") {
                CHECK(traverse.getCurrentVoxel()(0) == 0);
                CHECK(traverse.getCurrentVoxel()(1) == 0);
                CHECK(traverse.getCurrentVoxel()(2) == 0);
            }
            while (traverse.isInBoundingBox())
                traverse.updateTraverse();

            THEN("You leave the volume at (10, 9, 9)") {
                REQUIRE_FALSE(traverse.isInBoundingBox());
                REQUIRE(traverse.getCurrentVoxel()(0) == 10);
                REQUIRE(traverse.getCurrentVoxel()(1) == 9);
                REQUIRE(traverse.getCurrentVoxel()(2) == 9);
            }
        }
    }
}

SCENARIO("Check that the first step into the 2D Volume is correct")
{
    size_t dim = 2;
    size_t x = 5;
    size_t y = 5;
    IndexVector_t volumeDims(dim);
    volumeDims << x, y;

    RealVector_t ro(dim);
    RealVector_t rd(dim);

    GIVEN("A 5x5 volume with uniform scaling")
    {
        BoundingBox aabb(volumeDims);

        WHEN("The ray direction has the biggest value on the y axis")
        {
            ro << 0, 0;
            rd << 0.5, 0.7;
            rd.normalize();

            Ray r(ro, rd);
            TraverseAABB traverse(aabb, r);
            CHECK(traverse.isInBoundingBox());

            THEN("The traversal is initially at (0, 0)") {
                auto voxel = traverse.getCurrentVoxel();
                CHECK(voxel(0) == 0);
                CHECK(voxel(1) == 0);
            }

            traverse.updateTraverse();

            THEN("The first step is in y direction") {
                REQUIRE(traverse.isInBoundingBox());
                REQUIRE(traverse.getCurrentVoxel()(0) == 0);
                REQUIRE(traverse.getCurrentVoxel()(1) == 1);
            }
        }

        WHEN("The ray direction has the biggest value on the x axis")
        {
            ro << 0, 0;
            rd << 0.7, 0.5;
            rd.normalize();

            Ray r(ro, rd);
            TraverseAABB traverse(aabb, r);
            CHECK(traverse.isInBoundingBox());

            THEN("The traversal is initially at (0, 0)") {
                auto voxel = traverse.getCurrentVoxel();
                CHECK(voxel(0) == 0);
                CHECK(voxel(1) == 0);
            }

            traverse.updateTraverse();

            THEN("The first step is in y direction") {
                REQUIRE(traverse.isInBoundingBox());
                REQUIRE(traverse.getCurrentVoxel()(0) == 1);
                REQUIRE(traverse.getCurrentVoxel()(1) == 0);
            }
        }
    }
}

SCENARIO("Traverse_Volume_2D_EachPointIsTested")
{
    // setup
    size_t dim = 2;
    size_t x = 128;
    size_t y = 128;
    IndexVector_t volumeDims(dim);
    volumeDims << x, y;
    BoundingBox aabb(volumeDims);

    RealVector_t ro(dim);
    ro << -168.274, -143.397;
    RealVector_t rd(dim);
    rd << 0.761124909, 0.648605406;
    rd.normalize();
    Ray r(ro, rd);

    TraverseAABB traverse(aabb, r);
    CHECK(traverse.isInBoundingBox());

    size_t iter = 0;
    while (traverse.isInBoundingBox()) {
        RealVector_t voxel = traverse.getCurrentVoxel().template cast<real_t>();
        INFO( "Current Voxel: (" << voxel(0) << ", " << voxel(1) << ") in iter: " << iter);

        REQUIRE(intersect(voxel, r));
        traverse.updateTraverse();
        iter++;
    }
}

SCENARIO("Traversal through 2D volume should be equal to a ray voxel intersection for every voxel along the way")
{
    // TODO make this a stronger test, for first some "easy" direction (parallel ones)
    // TODO Then make some harder ones
    // setup
    size_t dim = 2;
    size_t x = 128;
    size_t y = 128;
    IndexVector_t volumeDims(dim);
    volumeDims << x, y;
    BoundingBox aabb(volumeDims);

    RealVector_t ro(dim);
    RealVector_t rd(dim);

    GIVEN("a point at the bottom left of the volume and a ray with leading dimension x") {
        ro << -168.274, -143.397;

        rd << 0.761124909, 0.648605406;
        rd.normalize();

        Ray r(ro, rd);

        THEN("Then all points the traversal visits are also hit by the intersection algorithm") {
            TraverseAABB traverse(aabb, r);
            CHECK(traverse.isInBoundingBox());

            size_t iter = 0;
            while (traverse.isInBoundingBox()) {
                RealVector_t voxel = traverse.getCurrentVoxel().template cast<real_t>();
                INFO( "Current Voxel: (" << voxel(0) << ", " << voxel(1) << ") in iter: " << iter);

                REQUIRE(intersect(voxel, r));
                traverse.updateTraverse();
                iter++;
            }
        }
    }

    GIVEN("a point at the bottom left of the volume and a ray with leading dimension y") {
        ro << 0, 0;

        rd << 0.648605406, 0.761124909;
        rd.normalize();

        Ray r(ro, rd);

        THEN("Then all points the traversal visits are also hit by the intersection algorithm") {
            TraverseAABB traverse(aabb, r);
            CHECK(traverse.isInBoundingBox());

            size_t iter = 0;
            while (traverse.isInBoundingBox()) {
                RealVector_t voxel = traverse.getCurrentVoxel().template cast<real_t>();
                INFO( "Current Voxel: (" << voxel(0) << ", " << voxel(1) << ") in iter: " << iter);

                REQUIRE(intersect(voxel, r));
                traverse.updateTraverse();
                iter++;
            }
        }
    }

    GIVEN("a ray going through the border of voxel column 0 and 1") {
        ro << 1, -0.5;

        rd << 0, 1;
        rd.normalize();

        Ray r(ro, rd);

        THEN("Then all points the traversal visits are also hit by the intersection algorithm") {
            TraverseAABB traverse(aabb, r);
            CHECK(traverse.isInBoundingBox());

            size_t iter = 0;
            while (traverse.isInBoundingBox()) {
                RealVector_t voxel = traverse.getCurrentVoxel().template cast<real_t>();
                INFO( "Current Voxel: (" << voxel(0) << ", " << voxel(1) << ") in iter: " << iter);

                REQUIRE(intersect(voxel, r));
                traverse.updateTraverse();
                iter++;
            }
        }
    }

    GIVEN("a ray going through the border of voxel row 0 and 1") {
        ro << -0.5, 1;

        rd << 1, 0;
        rd.normalize();

        Ray r(ro, rd);

        THEN("Then all points the traversal visits are also hit by the intersection algorithm") {
            TraverseAABB traverse(aabb, r);
            CHECK(traverse.isInBoundingBox());

            size_t iter = 0;
            while (traverse.isInBoundingBox()) {
                RealVector_t voxel = traverse.getCurrentVoxel().template cast<real_t>();
                INFO( "Current Voxel: (" << voxel(0) << ", " << voxel(1) << ") in iter: " << iter);

                REQUIRE(intersect(voxel, r));
                traverse.updateTraverse();
                iter++;
            }
        }
    }

    GIVEN("A ray going diagonally through the volume") {
        ro << -0.5, -0.5;

        rd << 1, 1;
        rd.normalize();

        Ray r(ro, rd);

        THEN("Then all points the traversal visits are also hit by the intersection algorithm") {
            TraverseAABB traverse(aabb, r);
            CHECK(traverse.isInBoundingBox());

            size_t iter = 0;
            while (traverse.isInBoundingBox()) {
                RealVector_t voxel = traverse.getCurrentVoxel().template cast<real_t>();
                INFO( "Current Voxel: (" << voxel(0) << ", " << voxel(1) << ") in iter: " << iter);

                REQUIRE(intersect(voxel, r));
                traverse.updateTraverse();
                iter++;
            }
        }
    }

    GIVEN("A ray going diagonally through the volume") {
        ro << -0.5, 32;

        rd << 1, 1;
        rd.normalize();

        Ray r(ro, rd);

        THEN("Then all points the traversal visits are also hit by the intersection algorithm") {
            TraverseAABB traverse(aabb, r);
            CHECK(traverse.isInBoundingBox());

            size_t iter = 0;
            while (traverse.isInBoundingBox()) {
                RealVector_t voxel = traverse.getCurrentVoxel().template cast<real_t>();
                INFO( "Current Voxel: (" << voxel(0) << ", " << voxel(1) << ") in iter: " << iter);

                REQUIRE(intersect(voxel, r));
                traverse.updateTraverse();
                iter++;
            }
        }
    }

    GIVEN("A ray going diagonally through the volume") {
        ro << -0.5, -0.5;

        rd << 0.699428, 0.472203;
        rd.normalize();

        Ray r(ro, rd);

        THEN("Then all points the traversal visits are also hit by the intersection algorithm") {
            TraverseAABB traverse(aabb, r);
            CHECK(traverse.isInBoundingBox());

            size_t iter = 0;
            while (traverse.isInBoundingBox()) {
                RealVector_t voxel = traverse.getCurrentVoxel().template cast<real_t>();
                INFO( "Current Voxel: (" << voxel(0) << ", " << voxel(1) << ") in iter: " << iter);

                REQUIRE(intersect(voxel, r));
                traverse.updateTraverse();
                iter++;
            }
        }
    }

    GIVEN("A ray going diagonally through the volume") {
        volumeDims << 64, 64;
        BoundingBox aabb(volumeDims);

        ro << 32.0002823, 3232;

        rd << -0.000723466626, -0.999999762;
        rd.normalize();

        Ray r(ro, rd);

        THEN("Then all points the traversal visits are also hit by the intersection algorithm") {
            TraverseAABB traverse(aabb, r);
            traverse.isInBoundingBox();

            size_t iter = 0;
            while (traverse.isInBoundingBox()) {
                RealVector_t voxel = traverse.getCurrentVoxel().template cast<real_t>();
                INFO( "Current Voxel: (" << voxel(0) << ", " << voxel(1) << ") in iter: " << iter);

                REQUIRE(intersect(voxel, r));
                traverse.updateTraverse();
                iter++;
            }
        }
    }
}
