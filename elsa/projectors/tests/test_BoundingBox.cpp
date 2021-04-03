/**
 * \file test_BoundingBox.cpp
 *
 * \brief Test for BoundingBox struct
 *
 * \author David Frank - initial code
 */

#include <catch2/catch.hpp>
#include "BoundingBox.h"

using namespace elsa;

SCENARIO("Testing 2d AABB")
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
                REQUIRE(aabb._dim == dim);

                REQUIRE(aabb._min(0) == 0);
                REQUIRE(aabb._min(1) == 0);

                REQUIRE(aabb._max(0) == (real_t) x);
                REQUIRE(aabb._max(1) == (real_t) y);

                REQUIRE(aabb._voxelCoordToIndexVector(0) == 1);
                REQUIRE(aabb._voxelCoordToIndexVector(1) == y);
            }
        }

        WHEN("copying the AABB")
        {
            BoundingBox aabb(volumeDims);
            auto aabbcopy = aabb;

            THEN("A copy is created succesfully")
            {
                REQUIRE(aabbcopy._dim == dim);

                REQUIRE(aabbcopy._min(0) == 0);
                REQUIRE(aabbcopy._min(1) == 0);

                REQUIRE(aabbcopy._max(0) == (real_t) x);
                REQUIRE(aabbcopy._max(1) == (real_t) y);

                REQUIRE(aabbcopy._voxelCoordToIndexVector(0) == 1);
                REQUIRE(aabbcopy._voxelCoordToIndexVector(1) == y);
            }
        }
    }
}

SCENARIO("Testing 3D aabb")
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
                REQUIRE(aabb._dim == dim);

                REQUIRE(aabb._min(0) == 0);
                REQUIRE(aabb._min(1) == 0);
                REQUIRE(aabb._min(2) == 0);

                REQUIRE(aabb._max(0) == (real_t) x);
                REQUIRE(aabb._max(1) == (real_t) y);
                REQUIRE(aabb._max(2) == (real_t) z);

                REQUIRE(aabb._voxelCoordToIndexVector(0) == 1);
                REQUIRE(aabb._voxelCoordToIndexVector(1) == y);
                REQUIRE(aabb._voxelCoordToIndexVector(2) == (z * z));
            }
        }
    }
}
