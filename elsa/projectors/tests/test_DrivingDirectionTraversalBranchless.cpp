
#include "doctest/doctest.h"

#include "DrivingDirectionTraversalBranchless.h"
#include "Intersection.h"

using namespace elsa;
using namespace doctest;

bool intersect(const RealVector_t& voxel, const RealRay_t& r)
{
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
    bb.min() += voxel;
    bb.max() += voxel;

    return intersectRay(bb, r).operator bool();
}

TEST_CASE("DrivingDirectionTraversalBranchless: Construction of a 2D traversal object")
{
    // setup
    const size_t dim = 2;
    index_t x = 3;
    index_t y = 3;
    IndexVector_t volumeDims(dim);
    volumeDims << x, y;
    IndexVector_t productOfCoefficientsPerDimension(2);
    productOfCoefficientsPerDimension << 1, 3;

    RealVector_t spacing(dim);

    RealVector_t ro(dim);
    RealVector_t rd(dim);

    GIVEN("A 3x3 aabb with standard spacing")
    {
        BoundingBox aabb(volumeDims);

        //================================
        //  intersection from straight rays from the bottom
        //================================
        WHEN("A traversal algorithms is initialised with the aabb and a ray with origin = (0.5, "
             "-0.5) and direction (0, 1)")
        {
            ro << 0.5, -0.5;
            rd << 0.0, 1.0;
            RealRay_t r(ro, rd);

            DrivingDirectionTraversalBranchless<dim> traverse(aabb, r);
            CHECK_UNARY(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel")
            {
                REQUIRE_EQ(traverse.getCurrentVoxel()(0), 0);
                REQUIRE_EQ(traverse.getCurrentVoxel()(1), 0);
            }
        }

        WHEN("A ray with origin = (1.0, -0.5) and direction (0, 1), hits the boundary between 2 "
             "voxels")
        {
            ro << 1.0, -0.5;
            rd << 0.0, 1.0;
            RealRay_t r(ro, rd);

            DrivingDirectionTraversalBranchless<dim> traverse(aabb, r);
            CHECK_UNARY(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel")
            {
                REQUIRE_EQ(traverse.getCurrentVoxel()(0), 1);
                REQUIRE_EQ(traverse.getCurrentVoxel()(1), 0);
            }
        }

        WHEN("A traversal algorithms is initialised with the aabb and a ray with origin = (1.5, "
             "-0.5) and direction (0, 1)")
        {
            ro << 1.5, -0.5;
            rd << 0.0, 1.0;
            RealRay_t r(ro, rd);

            DrivingDirectionTraversalBranchless<dim> traverse(aabb, r);
            CHECK_UNARY(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel")
            {
                REQUIRE_EQ(traverse.getCurrentVoxel()(0), 1);
                REQUIRE_EQ(traverse.getCurrentVoxel()(1), 0);
            }
        }

        WHEN("A ray with origin = (2.0, -0.5) and direction (0, 1), hits the boundary between 2 "
             "voxels")
        {
            ro << 2.0, -0.5;
            rd << 0.0, 1.0;
            RealRay_t r(ro, rd);

            DrivingDirectionTraversalBranchless<dim> traverse(aabb, r);
            CHECK_UNARY(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel")
            {
                REQUIRE_EQ(traverse.getCurrentVoxel()(0), 2);
                REQUIRE_EQ(traverse.getCurrentVoxel()(1), 0);
            }
        }

        WHEN("A traversal algorithms is initialised with the aabb and a ray with origin = (2.5, "
             "-0.5) and direction (0, 1)")
        {
            ro << 2.5, -0.5;
            rd << 0.0, 1.0;
            RealRay_t r(ro, rd);

            DrivingDirectionTraversalBranchless<dim> traverse(aabb, r);
            CHECK_UNARY(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel")
            {
                REQUIRE_EQ(traverse.getCurrentVoxel()(0), 2);
                REQUIRE_EQ(traverse.getCurrentVoxel()(1), 0);
            }
        }

        //================================
        //  intersection from straight rays from the left
        //================================
        WHEN("A traversal algorithms is initialised with the aabb and a ray with origin = (-0.5, "
             "0.5) and direction (1, 0)")
        {
            ro << -0.5, 0.5;
            rd << 1.0, 0.0;
            RealRay_t r(ro, rd);

            DrivingDirectionTraversalBranchless<dim> traverse(aabb, r);
            CHECK_UNARY(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel")
            {
                REQUIRE_EQ(traverse.getCurrentVoxel()(0), 0);
                REQUIRE_EQ(traverse.getCurrentVoxel()(1), 0);
            }
        }

        WHEN("A ray with origin = (1.0, -0.5) and direction (0, 1), hits the boundary between 2 "
             "voxels")
        {
            ro << -0.5, 1.0;
            rd << 1.0, 0.0;
            RealRay_t r(ro, rd);

            DrivingDirectionTraversalBranchless<dim> traverse(aabb, r);
            CHECK_UNARY(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel")
            {
                REQUIRE_EQ(traverse.getCurrentVoxel()(0), 0);
                REQUIRE_EQ(traverse.getCurrentVoxel()(1), 1);
            }
        }

        WHEN("A traversal algorithms is initialised with the aabb and a ray with origin = (-0.5, "
             "1.5) and direction (1, 0)")
        {
            ro << -0.5, 1.5;
            rd << 1.0, 0.0;
            RealRay_t r(ro, rd);

            DrivingDirectionTraversalBranchless<dim> traverse(aabb, r);
            CHECK_UNARY(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel")
            {
                REQUIRE_EQ(traverse.getCurrentVoxel()(0), 0);
                REQUIRE_EQ(traverse.getCurrentVoxel()(1), 1);
            }
        }

        WHEN("A ray with origin = (2.0, -0.5) and direction (0, 1), hits the boundary between 2 "
             "voxels")
        {
            ro << -0.5, 2.0;
            rd << 1.0, 0.0;
            RealRay_t r(ro, rd);

            DrivingDirectionTraversalBranchless<dim> traverse(aabb, r);
            CHECK_UNARY(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel")
            {
                REQUIRE_EQ(traverse.getCurrentVoxel()(0), 0);
                REQUIRE_EQ(traverse.getCurrentVoxel()(1), 2);
            }
        }

        WHEN("A traversal algorithms is initialised with the aabb and a ray with origin = (-0.5, "
             "2.5) and direction (1, 0)")
        {
            ro << -0.5, 2.5;
            rd << 1.0, 0.0;
            RealRay_t r(ro, rd);

            DrivingDirectionTraversalBranchless<dim> traverse(aabb, r);
            CHECK_UNARY(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel")
            {
                REQUIRE_EQ(traverse.getCurrentVoxel()(0), 0);
                REQUIRE_EQ(traverse.getCurrentVoxel()(1), 2);
            }
        }

        //================================
        //  intersection from straight rays from the right
        //================================
        WHEN("A traversal algorithms is initialised with the aabb and a ray with origin = (3.5, "
             "0.5) and direction (-1, 0)")
        {
            ro << 3.5, 0.5;
            rd << -1.0, 0.0;
            RealRay_t r(ro, rd);

            DrivingDirectionTraversalBranchless<dim> traverse(aabb, r);
            CHECK_UNARY(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel")
            {
                REQUIRE_EQ(traverse.getCurrentVoxel()(0), 2);
                REQUIRE_EQ(traverse.getCurrentVoxel()(1), 0);
            }
        }

        WHEN("A ray with origin = (3.5, 1.0) and direction (-1, 0), hits the boundary between 2 "
             "voxels")
        {
            ro << 3.5, 1.0;
            rd << -1.0, 0.0;
            RealRay_t r(ro, rd);

            DrivingDirectionTraversalBranchless<dim> traverse(aabb, r);
            CHECK_UNARY(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel")
            {
                REQUIRE_EQ(traverse.getCurrentVoxel()(0), 2);
                REQUIRE_EQ(traverse.getCurrentVoxel()(1), 1);
            }
        }

        WHEN("A traversal algorithms is initialised with the aabb and a ray with origin = (3.5, "
             "1.5) and direction (-1, 0)")
        {
            ro << 3.5, 1.5;
            rd << -1.0, 0.0;
            RealRay_t r(ro, rd);

            DrivingDirectionTraversalBranchless<dim> traverse(aabb, r);
            CHECK_UNARY(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel")
            {
                REQUIRE_EQ(traverse.getCurrentVoxel()(0), 2);
                REQUIRE_EQ(traverse.getCurrentVoxel()(1), 1);
            }
        }

        WHEN("A ray with origin = (3.5, 2.0) and direction (-1, 0), hits the boundary between 2 "
             "voxels")
        {
            ro << 3.5, 2.0;
            rd << -1.0, 0.0;
            RealRay_t r(ro, rd);

            DrivingDirectionTraversalBranchless<dim> traverse(aabb, r);
            CHECK_UNARY(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel")
            {
                REQUIRE_EQ(traverse.getCurrentVoxel()(0), 2);
                REQUIRE_EQ(traverse.getCurrentVoxel()(1), 2);
            }
        }

        WHEN("A traversal algorithms is initialised with the aabb and a ray with origin = (3.5, "
             "2.5) and direction (-1, 0)")
        {
            ro << 3.5, 2.5;
            rd << -1.0, 0.0;
            RealRay_t r(ro, rd);

            DrivingDirectionTraversalBranchless<dim> traverse(aabb, r);
            CHECK_UNARY(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel")
            {
                REQUIRE_EQ(traverse.getCurrentVoxel()(0), 2);
                REQUIRE_EQ(traverse.getCurrentVoxel()(1), 2);
            }
        }

        //================================
        //  intersection from straight rays from the top
        //================================
        WHEN("A traversal algorithms is initialised with the aabb and a ray with origin = (0.5, "
             "3.5) and direction (0, -1)")
        {
            ro << 0.5, 3.5;
            rd << 0.0, -1.0;
            RealRay_t r(ro, rd);

            DrivingDirectionTraversalBranchless<dim> traverse(aabb, r);
            CHECK_UNARY(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel")
            {
                REQUIRE_EQ(traverse.getCurrentVoxel()(0), 0);
                REQUIRE_EQ(traverse.getCurrentVoxel()(1), 2);
            }
        }

        WHEN("A ray with origin = (1.0, 3.5) and direction (-1, 0), hits the boundary between 2 "
             "voxels")
        {
            ro << 1.0, 3.5;
            rd << 0.0, -1.0;
            RealRay_t r(ro, rd);

            DrivingDirectionTraversalBranchless<dim> traverse(aabb, r);
            CHECK_UNARY(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel")
            {
                REQUIRE_EQ(traverse.getCurrentVoxel()(0), 1);
                REQUIRE_EQ(traverse.getCurrentVoxel()(1), 2);
            }
        }

        WHEN("A traversal algorithms is initialised with the aabb and a ray with origin = (1.5, "
             "3.5) and direction (0, -1)")
        {
            ro << 1.5, 3.5;
            rd << 0.0, -1.0;
            RealRay_t r(ro, rd);

            DrivingDirectionTraversalBranchless<dim> traverse(aabb, r);
            CHECK_UNARY(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel")
            {
                REQUIRE_EQ(traverse.getCurrentVoxel()(0), 1);
                REQUIRE_EQ(traverse.getCurrentVoxel()(1), 2);
            }
        }

        WHEN("A ray with origin = (2.0, 3.5) and direction (-1, 0), hits the boundary between 2 "
             "voxels")
        {
            ro << 2.0, 3.5;
            rd << 0.0, -1.0;
            RealRay_t r(ro, rd);

            DrivingDirectionTraversalBranchless<dim> traverse(aabb, r);
            CHECK_UNARY(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel")
            {
                REQUIRE_EQ(traverse.getCurrentVoxel()(0), 2);
                REQUIRE_EQ(traverse.getCurrentVoxel()(1), 2);
            }
        }

        WHEN("A traversal algorithms is initialised with the aabb and a ray with origin = (2.5, "
             "3.5) and direction (0, -1)")
        {
            ro << 2.5, 3.5;
            rd << 0.0, -1.0;
            RealRay_t r(ro, rd);

            DrivingDirectionTraversalBranchless<dim> traverse(aabb, r);
            CHECK_UNARY(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel")
            {
                REQUIRE_EQ(traverse.getCurrentVoxel()(0), 2);
                REQUIRE_EQ(traverse.getCurrentVoxel()(1), 2);
            }
        }

        //
        // Some edge cases
        //
        WHEN("A ray with origin = (-0.5, 0.0) and direction (1, 0) hits the left edge of aabb")
        {
            ro << -0.5, 0.0;
            rd << 1.0, 0.0;
            RealRay_t r(ro, rd);

            DrivingDirectionTraversalBranchless<dim> traverse(aabb, r);
            CHECK_UNARY(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel")
            {
                REQUIRE_EQ(traverse.getCurrentVoxel()(0), 0);
                REQUIRE_EQ(traverse.getCurrentVoxel()(1), 0);
            }
        }

        WHEN("A ray with origin = (3.5, 0.0) and direction (-1, 0) hits the left edge of aabb")
        {
            ro << 3.5, 0.0;
            rd << -1.0, 0.0;
            RealRay_t r(ro, rd);

            DrivingDirectionTraversalBranchless<dim> traverse(aabb, r);
            CHECK_UNARY(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the top left pixel")
            {
                REQUIRE_EQ(traverse.getCurrentVoxel()(0), 2);
                REQUIRE_EQ(traverse.getCurrentVoxel()(1), 0);
            }
        }

        WHEN("A ray with origin = (0.0, -0.5) and direction (0, 1) hits the bottom edge of aabb)")
        {
            ro << 0.0, -0.5;
            rd << 0.0, 1.0;
            RealRay_t r(ro, rd);

            DrivingDirectionTraversalBranchless<dim> traverse(aabb, r);
            CHECK_UNARY(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the bottom left pixel")
            {
                REQUIRE_EQ(traverse.getCurrentVoxel()(0), 0);
                REQUIRE_EQ(traverse.getCurrentVoxel()(1), 0);
            }
        }

        WHEN("A ray with origin = (0.0, 3.5) and direction (0, -1) hits the top edge of aabb)")
        {
            ro << 0.0, 3.5;
            rd << 0.0, -1.0;
            RealRay_t r(ro, rd);

            DrivingDirectionTraversalBranchless<dim> traverse(aabb, r);
            CHECK_UNARY(traverse.isInBoundingBox());
            THEN("The ray intersects the aabb at the top left pixel")
            {
                REQUIRE_EQ(traverse.getCurrentVoxel()(0), 0);
                REQUIRE_EQ(traverse.getCurrentVoxel()(1), 2);
            }
        }
    }
}

TEST_CASE("DrivingDirectionTraversalBranchless: Traverse a 2D volume and only check that the "
          "endpoint is correct")
{
    // setup
    const size_t dim = 2;
    index_t x = 10;
    index_t y = 10;
    IndexVector_t volumeDims(dim);
    volumeDims << x, y;
    IndexVector_t productOfCoefficientsPerDimension(2);
    productOfCoefficientsPerDimension << 1, 10;

    RealVector_t spacing(dim);
    RealVector_t ro(dim);
    RealVector_t rd(dim);

    GIVEN("A 10x10 volume with uniform scaling")
    {
        BoundingBox aabb(volumeDims);

        WHEN("The volume is traversed with a ray with origin = (-0.5, 0.5, 0.5) and a direction = "
             "(0, 1, 0)")
        {
            ro << -1, 4.5;
            rd << 1.0, 0;

            RealRay_t r(ro, rd);
            DrivingDirectionTraversalBranchless<dim> traverse(aabb, r);
            CHECK_UNARY(traverse.isInBoundingBox());

            while (traverse.isInBoundingBox())
                traverse.updateTraverse();

            THEN("The endpoint should be (10,4)")
            {
                REQUIRE_UNARY_FALSE(traverse.isInBoundingBox());
                REQUIRE_EQ(traverse.getCurrentVoxel()(0), 10);
                REQUIRE_EQ(traverse.getCurrentVoxel()(1), 4);
            }
        }

        WHEN("The volume is traversed with a ray with origin = (-1, -1) and a direction = "
             "(1, 1)")
        {
            ro << -1, -1;
            rd << 1.0, 1.0;
            rd.normalize();

            RealRay_t r(ro, rd);
            DrivingDirectionTraversalBranchless<dim> traverse(aabb, r);
            CHECK_UNARY(traverse.isInBoundingBox());

            while (traverse.isInBoundingBox())
                traverse.updateTraverse();

            THEN("The endpoint should be (10,10)")
            {
                REQUIRE_UNARY_FALSE(traverse.isInBoundingBox());
                REQUIRE_EQ(traverse.getCurrentVoxel()(0), 10);
                REQUIRE_EQ(traverse.getCurrentVoxel()(1), 10);
            }
        }
    }
}

TEST_CASE(
    "DrivingDirectionTraversalBranchless: Check that the first step into the 2D Volume is correct")
{
    const size_t dim = 2;
    index_t x = 5;
    index_t y = 5;
    IndexVector_t volumeDims(dim);
    volumeDims << x, y;
    IndexVector_t productOfCoefficientsPerDimension(2);
    productOfCoefficientsPerDimension << 1, 5;

    RealVector_t ro(dim);
    RealVector_t rd(dim);

    GIVEN("A 5x5 volume with uniform scaling")
    {
        BoundingBox aabb(volumeDims);

        WHEN("The ray direction has the biggest value on the y axis")
        {
            ro << 0, 0;
            rd << 0.5f, 0.7f;
            rd.normalize();

            RealRay_t r(ro, rd);
            DrivingDirectionTraversalBranchless<dim> traverse(aabb, r);
            CHECK_UNARY(traverse.isInBoundingBox());

            THEN("The traversal is initially at (0, 0)")
            {
                auto voxel = traverse.getCurrentVoxel();
                CHECK_EQ(voxel(0), 0);
                CHECK_EQ(voxel(1), 0);
            }

            traverse.updateTraverse();

            THEN("The first step is in y direction")
            {
                REQUIRE_UNARY(traverse.isInBoundingBox());
                REQUIRE_EQ(traverse.getCurrentVoxel()(0), 0);
                REQUIRE_EQ(traverse.getCurrentVoxel()(1), 1);
            }
        }

        WHEN("The ray direction has the biggest value on the x axis")
        {
            ro << 0, 0;
            rd << 0.7f, 0.5f;
            rd.normalize();

            RealRay_t r(ro, rd);
            DrivingDirectionTraversalBranchless<dim> traverse(aabb, r);
            CHECK_UNARY(traverse.isInBoundingBox());

            THEN("The traversal is initially at (0, 0)")
            {
                auto voxel = traverse.getCurrentVoxel();
                CHECK_EQ(voxel(0), 0);
                CHECK_EQ(voxel(1), 0);
            }

            traverse.updateTraverse();

            THEN("The first step is in y direction")
            {
                REQUIRE_UNARY(traverse.isInBoundingBox());
                REQUIRE_EQ(traverse.getCurrentVoxel()(0), 1);
                REQUIRE_EQ(traverse.getCurrentVoxel()(1), 0);
            }
        }
    }
}

TEST_CASE("DrivingDirectionTraversalBranchless: Traverse_Volume_2D_EachPointIsTested")
{
    // setup
    const size_t dim = 2;
    index_t x = 128;
    index_t y = 128;
    IndexVector_t volumeDims(dim);
    volumeDims << x, y;
    BoundingBox aabb(volumeDims);
    IndexVector_t productOfCoefficientsPerDimension(2);
    productOfCoefficientsPerDimension << 1, 128;

    RealVector_t ro(dim);
    ro << -168.274f, -143.397f;
    RealVector_t rd(dim);
    rd << 0.761124909f, 0.648605406f;
    rd.normalize();
    RealRay_t r(ro, rd);

    DrivingDirectionTraversalBranchless<dim> traverse(aabb, r);
    CHECK_UNARY(traverse.isInBoundingBox());

    size_t iter = 0;
    while (traverse.isInBoundingBox()) {
        RealVector_t voxel = traverse.getCurrentVoxel().template cast<real_t>();
        INFO("Current Voxel: (" << voxel(0) << ", " << voxel(1) << ") in iter: " << iter);

        REQUIRE_UNARY(intersect(voxel, r));
        traverse.updateTraverse();
        iter++;
    }
}

TEST_CASE("DrivingDirectionTraversalBranchless: Traversal through 2D volume should be equal to a "
          "ray voxel intersection "
          "for every voxel along the way")
{
    // TODO make this a stronger test, for first some "easy" direction (parallel ones)
    // TODO Then make some harder ones
    // setup
    const size_t dim = 2;
    index_t x = 128;
    index_t y = 128;
    IndexVector_t volumeDims(dim);
    volumeDims << x, y;
    BoundingBox aabb(volumeDims);
    IndexVector_t productOfCoefficientsPerDimension(2);
    productOfCoefficientsPerDimension << 1, 128;

    RealVector_t ro(dim);
    RealVector_t rd(dim);

    GIVEN("a point at the bottom left of the volume and a ray with leading dimension x")
    {
        ro << -168.274f, -143.397f;

        rd << 0.761124909f, 0.648605406f;
        rd.normalize();

        RealRay_t r(ro, rd);

        THEN("Then all points the traversal visits are also hit by the intersection algorithm")
        {
            DrivingDirectionTraversalBranchless<dim> traverse(aabb, r);
            CHECK_UNARY(traverse.isInBoundingBox());

            size_t iter = 0;
            while (traverse.isInBoundingBox()) {
                RealVector_t voxel = traverse.getCurrentVoxel().template cast<real_t>();
                INFO("Current Voxel: (" << voxel(0) << ", " << voxel(1) << ") in iter: " << iter);

                REQUIRE_UNARY(intersect(voxel, r));
                traverse.updateTraverse();
                iter++;
            }
        }
    }

    GIVEN("a point at the bottom left of the volume and a ray with leading dimension y")
    {
        ro << 0, 0;

        rd << 0.648605406f, 0.761124909f;
        rd.normalize();

        RealRay_t r(ro, rd);

        THEN("Then all points the traversal visits are also hit by the intersection algorithm")
        {
            DrivingDirectionTraversalBranchless<dim> traverse(aabb, r);
            CHECK_UNARY(traverse.isInBoundingBox());

            size_t iter = 0;
            while (traverse.isInBoundingBox()) {
                RealVector_t voxel = traverse.getCurrentVoxel().template cast<real_t>();
                INFO("Current Voxel: (" << voxel(0) << ", " << voxel(1) << ") in iter: " << iter);

                REQUIRE_UNARY(intersect(voxel, r));
                traverse.updateTraverse();
                iter++;
            }
        }
    }

    GIVEN("a ray going through the border of voxel column 0 and 1")
    {
        ro << 1, -0.5f;

        rd << 0, 1;
        rd.normalize();

        RealRay_t r(ro, rd);

        THEN("Then all points the traversal visits are also hit by the intersection algorithm")
        {
            DrivingDirectionTraversalBranchless<dim> traverse(aabb, r);
            CHECK_UNARY(traverse.isInBoundingBox());

            size_t iter = 0;
            while (traverse.isInBoundingBox()) {
                RealVector_t voxel = traverse.getCurrentVoxel().template cast<real_t>();
                INFO("Current Voxel: (" << voxel(0) << ", " << voxel(1) << ") in iter: " << iter);

                REQUIRE_UNARY(intersect(voxel, r));
                traverse.updateTraverse();
                iter++;
            }
        }
    }

    GIVEN("a ray going through the border of voxel row 0 and 1")
    {
        ro << -0.5f, 1;

        rd << 1, 0;
        rd.normalize();

        RealRay_t r(ro, rd);

        THEN("Then all points the traversal visits are also hit by the intersection algorithm")
        {
            DrivingDirectionTraversalBranchless<dim> traverse(aabb, r);
            CHECK_UNARY(traverse.isInBoundingBox());

            size_t iter = 0;
            while (traverse.isInBoundingBox()) {
                RealVector_t voxel = traverse.getCurrentVoxel().template cast<real_t>();
                INFO("Current Voxel: (" << voxel(0) << ", " << voxel(1) << ") in iter: " << iter);

                REQUIRE_UNARY(intersect(voxel, r));
                traverse.updateTraverse();
                iter++;
            }
        }
    }

    GIVEN("A ray going diagonally through the volume")
    {
        ro << -0.5f, -0.5f;

        rd << 1, 1;
        rd.normalize();

        RealRay_t r(ro, rd);

        THEN("Then all points the traversal visits are also hit by the intersection algorithm")
        {
            DrivingDirectionTraversalBranchless<dim> traverse(aabb, r);
            CHECK_UNARY(traverse.isInBoundingBox());

            size_t iter = 0;
            while (traverse.isInBoundingBox()) {
                RealVector_t voxel = traverse.getCurrentVoxel().template cast<real_t>();
                INFO("Current Voxel: (" << voxel(0) << ", " << voxel(1) << ") in iter: " << iter);

                REQUIRE_UNARY(intersect(voxel, r));
                traverse.updateTraverse();
                iter++;
            }
        }
    }

    GIVEN("A ray going diagonally through the volume")
    {
        ro << -0.5f, 32;

        rd << 1, 1;
        rd.normalize();

        RealRay_t r(ro, rd);

        THEN("Then all points the traversal visits are also hit by the intersection algorithm")
        {
            DrivingDirectionTraversalBranchless<dim> traverse(aabb, r);
            CHECK_UNARY(traverse.isInBoundingBox());

            size_t iter = 0;
            while (traverse.isInBoundingBox()) {
                RealVector_t voxel = traverse.getCurrentVoxel().template cast<real_t>();
                INFO("Current Voxel: (" << voxel(0) << ", " << voxel(1) << ") in iter: " << iter);

                REQUIRE_UNARY(intersect(voxel, r));
                traverse.updateTraverse();
                iter++;
            }
        }
    }

    GIVEN("A ray going diagonally through the volume")
    {
        ro << -0.5f, -0.5f;

        rd << 0.699428f, 0.472203f;
        rd.normalize();

        RealRay_t r(ro, rd);

        THEN("Then all points the traversal visits are also hit by the intersection algorithm")
        {
            DrivingDirectionTraversalBranchless<dim> traverse(aabb, r);
            CHECK_UNARY(traverse.isInBoundingBox());

            size_t iter = 0;
            while (traverse.isInBoundingBox()) {
                RealVector_t voxel = traverse.getCurrentVoxel().template cast<real_t>();
                INFO("Current Voxel: (" << voxel(0) << ", " << voxel(1) << ") in iter: " << iter);

                REQUIRE_UNARY(intersect(voxel, r));
                traverse.updateTraverse();
                iter++;
            }
        }
    }

    GIVEN("A ray going diagonally through the volume")
    {
        volumeDims << 64, 64;
        BoundingBox aabb(volumeDims);

        ro << 32.0002823f, 3232;

        rd << -0.000723466626f, -0.999999762f;
        rd.normalize();

        RealRay_t r(ro, rd);

        THEN("Then all points the traversal visits are also hit by the intersection algorithm")
        {
            DrivingDirectionTraversalBranchless<dim> traverse(aabb, r);
            traverse.isInBoundingBox();

            size_t iter = 0;
            while (traverse.isInBoundingBox()) {
                RealVector_t voxel = traverse.getCurrentVoxel().template cast<real_t>();
                INFO("Current Voxel: (" << voxel(0) << ", " << voxel(1) << ") in iter: " << iter);

                REQUIRE_UNARY(intersect(voxel, r));
                traverse.updateTraverse();
                iter++;
            }
        }
    }
}
