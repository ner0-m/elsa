/**
 * @file test_HardThresholding.cpp
 *
 * @brief Tests for the HardThresholding class
 *
 * @author Andi Braimllari
 */

#include "Error.h"
#include "HardThresholding.h"
#include "ProximityOperator.h"
#include "VolumeDescriptor.h"

#include "doctest/doctest.h"
#include "elsaDefines.h"
#include <testHelpers.h>

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("proximity_operators");

TEST_CASE_TEMPLATE("SoftThresholding: Testing regularity", data_t, float, double)
{
    static_assert(std::is_default_constructible_v<HardThresholding<data_t>>);
    static_assert(std::is_copy_assignable_v<HardThresholding<data_t>>);
    static_assert(std::is_copy_constructible_v<HardThresholding<data_t>>);
    static_assert(std::is_nothrow_move_assignable_v<HardThresholding<data_t>>);
    static_assert(std::is_nothrow_move_constructible_v<HardThresholding<data_t>>);
}

TEST_CASE_TEMPLATE("SoftThresholding: Testing in 1D", data_t, float, double)
{
    GIVEN("a DataDescriptor")
    {
        IndexVector_t numCoeff({{8}});
        VolumeDescriptor volDescr(numCoeff);

        WHEN("Using HardThresholding operator in 1D")
        {
            HardThresholding<data_t> hardThrsOp;

            THEN("Values under threshold=4 are 0")
            {
                DataContainer<data_t> x(volDescr, Vector_t<data_t>({{-2, 3, 4, -7, 7, 8, 8, 3}}));
                DataContainer<data_t> expected(volDescr,
                                               Vector_t<data_t>({{0, 0, 0, -7, 7, 8, 8, 0}}));

                auto res = hardThrsOp.apply(x, geometry::Threshold<data_t>{4});
                REQUIRE_UNARY(isApprox(expected, res));
            }

            THEN("Is works when accessed as ProximityOperator")
            {
                ProximityOperator<data_t> prox(hardThrsOp);

                DataContainer<data_t> x(volDescr, Vector_t<data_t>({{-2, 3, 4, -7, 7, 8, 8, 3}}));
                DataContainer<data_t> expected(volDescr,
                                               Vector_t<data_t>({{0, 0, 0, -7, 7, 8, 8, 0}}));

                auto res = prox.apply(x, geometry::Threshold<data_t>{4});
                REQUIRE_UNARY(isApprox(expected, res));
            }
        }
    }
}

TEST_CASE_TEMPLATE("HardThresholding: Testing in 3D", data_t, float, double)
{
    GIVEN("a DataDescriptor")
    {
        IndexVector_t numCoeff({{3, 2, 3}});
        VolumeDescriptor volumeDescriptor(numCoeff);

        WHEN("Using HardThresholding operator in 3D")
        {
            HardThresholding<data_t> hardThrsOp;

            THEN("Values under threshold=5 are 0 and values above remain the same")
            {
                DataContainer<data_t> x(
                    volumeDescriptor,
                    Vector_t<data_t>({{2, 1, 6, 6, 1, 4, 2, -9, 7, 7, 7, 3, 1, 2, 8, 9, -4, 5}}));

                DataContainer<data_t> expected(
                    volumeDescriptor,
                    Vector_t<data_t>{{{0, 0, 6, 6, 0, 0, 0, -9, 7, 7, 7, 0, 0, 0, 8, 9, 0, 0}}});

                auto res = hardThrsOp.apply(x, geometry::Threshold<data_t>{5});
                REQUIRE_UNARY(isApprox(expected, res));
            }
        }
    }
}

TEST_CASE_TEMPLATE("HardThresholding: Testing general behaviour", data_t, float, double)
{
    GIVEN("a DataDescriptor")
    {
        IndexVector_t numCoeff({{8}});
        VolumeDescriptor desc(numCoeff);

        WHEN("Using HardThresholding operator")
        {
            HardThresholding<data_t> hardThrOp;

            THEN("The zero vector is returned when the zero vector is given")
            {
                auto zero = Vector_t<data_t>{{{0, 0, 0, 0, 0, 0, 0, 0}}};
                DataContainer<data_t> x(desc, zero);
                DataContainer<data_t> expected(desc, zero);

                auto res = hardThrOp.apply(x, geometry::Threshold<data_t>{4});
                REQUIRE_UNARY(isApprox(expected, res));
            }

            THEN("HardThresholding operator throws exception for t <= 0")
            {
                DataContainer<data_t> x(desc, Vector_t<data_t>{{{0, 0, 0, 0, 0, 0, 0, 0}}});

                // actually the geometry::Threshold throws this
                REQUIRE_THROWS_AS(hardThrOp.apply(x, geometry::Threshold<data_t>{0}),
                                  InvalidArgumentError);

                // actually the geometry::Threshold throws this
                REQUIRE_THROWS_AS(hardThrOp.apply(x, geometry::Threshold<data_t>{-1}),
                                  InvalidArgumentError);
            }

            THEN("HardThresholding operator throws exception for differently sized v and prox")
            {
                DataContainer<data_t> x(desc, Vector_t<data_t>{{{0, 0, 0, 0, 0, 0, 0, 0}}});

                VolumeDescriptor largeDesc(IndexVector_t{{9}});
                DataContainer<data_t> largeX(largeDesc,
                                             Vector_t<data_t>{{{0, 0, 0, 0, 0, 0, 0, 0, 0}}});

                REQUIRE_THROWS_AS(hardThrOp.apply(x, geometry::Threshold<data_t>{1}, largeX),
                                  LogicError);
            }
        }
    }
}

TEST_SUITE_END();
