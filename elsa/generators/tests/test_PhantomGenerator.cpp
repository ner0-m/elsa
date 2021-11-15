/**
 * @file test_PhantomGenerator.cpp
 *
 * @brief Tests for the PhantomGenerator class
 *
 * @author Tobias Lasser - nothing to see here...
 */

#include "doctest/doctest.h"
#include "PhantomGenerator.h"
#include "VolumeDescriptor.h"
#include "testHelpers.h"
#include <array>
#include <iostream>

using namespace elsa;
using namespace doctest;

RealVector_t get2dModifiedSheppLogan45x45();

TEST_CASE_TEMPLATE("PhantomGenerator: Drawing a 2d Shepp-Logan phantom", data_t, float, double)
{

    GIVEN("A small 2D volume")
    {
        const IndexVector_t size({{16, 16}});

        WHEN("Creating the Sheep Logan phantom")
        {
            const auto dc = PhantomGenerator<data_t>::createModifiedSheppLogan(size);

            THEN("It's close to the reference (This is just to track difference)")
            {
// I'm sorry, but I'm not going to cast each and every single float here
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wimplicit-float-conversion"
                const Vector_t<data_t> expected({{
                    // clang-format off
                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                    0,   0,   0,   0,   0,   0,   0, 0.2, 0.2, 0.2,   0,   0,   0,   0,   0,   0,
                    0,   0,   0,   0,   0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,   0,   0,   0,   0,
                    0,   0,   0,   0,   0, 0.2, 0.2, 0.3, 0.3, 0.3, 0.2, 0.2,   0,   0,   0,   0,
                    0,   0,   0,   0, 0.2,   0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2,   0,   0,   0,
                    0,   0,   0,   0, 0.2,   0, 0.1, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2,   0,   0,   0,
                    0,   0,   0, 0.2, 0.2,   0, 0.1, 0.3, 0.3, 0.3, 0.1, 0.2, 0.2, 0.2,   0,   0,
                    0,   0,   0, 0.2, 0.2,   0,   0, 0.3, 0.3, 0.1,   0,   0, 0.2, 0.2,   0,   0,
                    0,   0,   0, 0.2, 0.2,   0,   0,   0, 0.2,   0,   0,   0, 0.2, 0.2,   0,   0,
                    0,   0,   0, 0.2, 0.2, 0.2,   0,   0, 0.2,   0,   0,   0, 0.2, 0.2,   0,   0,
                    0,   0,   0, 0.2, 0.2, 0.2,   0,   0, 0.2, 0.2,   0, 0.2, 0.2, 0.2,   0,   0,
                    0,   0,   0,   0, 0.2, 0.2,   0,   0, 0.2, 0.2, 0.2, 0.2, 0.2,   0,   0,   0,
                    0,   0,   0,   0, 0.2, 0.2, 0.2,   0, 0.2, 0.2, 0.2, 0.2, 0.2,   0,   0,   0,
                    0,   0,   0,   0,   0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,   0,   0,   0,   0,
                    0,   0,   0,   0,   0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,   0,   0,   0,   0,
                    0,   0,   0,   0,   0,   0,   0, 0.2, 0.2, 0.2,   0,   0,   0,   0,   0,   0
                    // clang-format on
                }});
#pragma GCC diagnostic pop

                auto ref = DataContainer(VolumeDescriptor(size), expected);

                INFO("Computed phantom: ", dc);
                INFO("Refernce phantom: ", ref);

                for (int i = 0; i < ref.getSize(); ++i) {
                    INFO("Error at position: ", i);
                    CHECK_UNARY(checkApproxEq(dc[i], ref[i]));
                }
            }
            THEN("It's not close to the matlab reference :-(")
            {
// I'm sorry, but I'm not going to cast each and every single float here
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wimplicit-float-conversion"
                const Vector_t<data_t> matlab({{
                    // clang-format off
                    0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  0,
                    0,  0,  0,   0,   0,   0,   1,   1,   1,   1,   0,   0,   0,  0,  0,  0,
                    0,  0,  0,   0,   0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,   0,   0,  0,  0,  0,
                    0,  0,  0,   0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,   0,  0,  0,  0,
                    0,  0,  0,   0, 0.2, 0.2, 0.2, 0.3, 0.3, 0.2, 0.2, 0.2,   0,  0,  0,  0,
                    0,  0,  0, 0.2, 0.2,   0, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2,  0,  0,  0,
                    0,  0,  0, 0.2, 0.2,   0,   0, 0.3, 0.3,   0,   0, 0.2, 0.2,  0,  0,  0,
                    0,  0,  0, 0.2, 0.2,   0,   0, 0.2, 0.2,   0,   0, 0.2, 0.2,  0,  0,  0,
                    0,  0,  0, 0.2, 0.2,   0,   0,   0, 0.2,   0, 0.2, 0.2, 0.2,  0,  0,  0,
                    0,  0,  0, 0.2, 0.2, 0.2,   0,   0, 0.2,   0, 0.2, 0.2, 0.2,  0,  0,  0,
                    0,  0,  0, 0.2, 0.2, 0.2,   0,   0, 0.2, 0.2, 0.2, 0.2, 0.2,  0,  0,  0,
                    0,  0,  0,   0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,   0,  0,  0,  0,
                    0,  0,  0,   0, 0.2, 0.2, 0.2, 0.3, 0.3, 0.2, 0.2, 0.2,   0,  0,  0,  0,
                    0,  0,  0,   0,   0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,   0,   0,  0,  0,  0,
                    0,  0,  0,   0,   0,   0,   1, 0.2, 0.2,   1,   0,   0,   0,  0,  0,  0,
                    0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  0
                    // clang-format on
                }});
#pragma GCC diagnostic pop

                auto ref = DataContainer(VolumeDescriptor(size), matlab);
                CHECK_UNARY_FALSE(isApprox(dc, ref));
            }
        }
    }
}

TEST_CASE("PhantomGenerator: Drawing a 3d Shepp-Logan phantom")
{
    GIVEN("a volume size")
    {
        IndexVector_t numCoeff({{64, 64, 64}});

        WHEN("creating a 3d Shepp-Logan")
        {
            auto dc = PhantomGenerator<real_t>::createModifiedSheppLogan(numCoeff);

            THEN("it looks good")
            {
                REQUIRE(true); // TODO: add a proper test here
            }
        }
    }
}
