/**
 * @file test_BoundingBox.cpp
 *
 * @brief Test for BoundingBox struct
 *
 * @author David Frank - initial code
 */

#include "doctest/doctest.h"
#include "BoundingBox.h"
#include "testHelpers.h"

using namespace elsa;
using namespace doctest;

TEST_CASE("BoundingBox: Testing 2D AABB")
{
    GIVEN("An aabb of size 10x15")
    {
        // setup
        std::size_t dim = 2;
        index_t x = 10;
        index_t y = 15;
        IndexVector_t volumeDims(dim);
        volumeDims << x, y;

        WHEN("instantiating the AABB")
        {
            BoundingBox aabb(volumeDims);

            THEN("The min is set to the origin (0, 0) and the max is (10, 15)")
            {
                REQUIRE_EQ(aabb.dim(), dim);

                REQUIRE_UNARY(checkApproxEq(aabb.min()(0), 0));
                REQUIRE_UNARY(checkApproxEq(aabb.min()(1), 0));

                REQUIRE_UNARY(checkApproxEq(aabb.max()(0), (real_t) x));
                REQUIRE_UNARY(checkApproxEq(aabb.max()(1), (real_t) y));
            }
        }

        WHEN("copying the AABB")
        {
            BoundingBox aabb(volumeDims);
            auto aabbcopy = aabb;

            THEN("A copy is created successfully")
            {
                REQUIRE_EQ(aabbcopy.dim(), dim);

                REQUIRE_UNARY(checkApproxEq(aabbcopy.min()(0), 0));
                REQUIRE_UNARY(checkApproxEq(aabbcopy.min()(1), 0));

                REQUIRE_UNARY(checkApproxEq(aabbcopy.max()(0), (real_t) x));
                REQUIRE_UNARY(checkApproxEq(aabbcopy.max()(1), (real_t) y));
            }
        }
    }
}

TEST_CASE("BoundingBox: Testing 3D aabb")
{
    GIVEN("An aabb of size 10x15x20 with uniform")
    {
        // setup
        size_t dim = 3;
        index_t x = 10;
        index_t y = 15;
        index_t z = 20;
        IndexVector_t volumeDims(dim);
        volumeDims << x, y, z;

        WHEN("instantiating the AABB")
        {
            BoundingBox aabb(volumeDims);

            THEN("The min is set to the origin (0, 0, 0) and the max is (10, 15, 20)")
            {
                REQUIRE_EQ(aabb.dim(), dim);

                REQUIRE_UNARY(checkApproxEq(aabb.min()(0), 0));
                REQUIRE_UNARY(checkApproxEq(aabb.min()(1), 0));
                REQUIRE_UNARY(checkApproxEq(aabb.min()(2), 0));

                REQUIRE_UNARY(checkApproxEq(aabb.max()(0), (real_t) x));
                REQUIRE_UNARY(checkApproxEq(aabb.max()(1), (real_t) y));
                REQUIRE_UNARY(checkApproxEq(aabb.max()(2), (real_t) z));
            }
        }
    }
}
