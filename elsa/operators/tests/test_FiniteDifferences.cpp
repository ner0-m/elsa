/**
 * @file test_FiniteDifferences.cpp
 *
 * @brief Tests for FiniteDifferences class
 *
 * @author Matthias Wieczorek - main code
 * @author Tobias Lasser - rewrite
 */

#include <catch2/catch.hpp>
#include "FiniteDifferences.h"
#include "VolumeDescriptor.h"

using namespace elsa;

SCENARIO("Constructing a FiniteDifferences operator")
{
    GIVEN("a descriptor")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 143, 48;
        VolumeDescriptor dd(numCoeff);

        IndexVector_t rangeCoeffs(3);
        rangeCoeffs << 143, 48, 2;
        VolumeDescriptor ddRange(rangeCoeffs);

        WHEN("instantiating a FiniteDifferences operator")
        {
            FiniteDifferences fdOp(dd);

            THEN("the descriptors are as expected")
            {
                REQUIRE(fdOp.getDomainDescriptor() == dd);
                REQUIRE(fdOp.getRangeDescriptor() == ddRange);
            }
        }

        WHEN("cloning a FiniteDifference operator")
        {
            FiniteDifferences fdOp(dd);
            auto fdOpClone = fdOp.clone();

            THEN("everything matches")
            {
                REQUIRE(fdOpClone.get() != &fdOp);
                REQUIRE(*fdOpClone == fdOp);
            }
        }
    }
}

SCENARIO("Testing FiniteDifferences in 1D")
{
    GIVEN("some data")
    {
        IndexVector_t numCoeff(1);
        numCoeff << 5;
        VolumeDescriptor dd(numCoeff);
        RealVector_t data(dd.getNumberOfCoefficients());
        data << 30, 3, 2, -1, 7;

        DataContainer dc(dd, data);

        WHEN("using forward differences (default mode)")
        {
            FiniteDifferences fdOp(dd);

            THEN("the results are correct")
            {
                RealVector_t resApply(fdOp.getRangeDescriptor().getNumberOfCoefficients());
                resApply << -27, -1, -3, 8, -7;
                DataContainer dcResApply(fdOp.getRangeDescriptor(), resApply);

                REQUIRE(dcResApply == fdOp.apply(dc));

                RealVector_t resApplyAdjoint(fdOp.getDomainDescriptor().getNumberOfCoefficients());
                resApplyAdjoint << 27, -26, 2, -11, 15;
                DataContainer dcResApplyAdjoint(fdOp.getDomainDescriptor(), resApplyAdjoint);

                REQUIRE(dcResApplyAdjoint == fdOp.applyAdjoint(dcResApply));
            }
        }

        WHEN("using backward differences")
        {
            FiniteDifferences fdOp(dd, FiniteDifferences<real_t>::DiffType::BACKWARD);

            THEN("the results are correct")
            {
                RealVector_t resApply(fdOp.getRangeDescriptor().getNumberOfCoefficients());
                resApply << 30, -27, -1, -3, 8;
                DataContainer dcResApply(fdOp.getRangeDescriptor(), resApply);

                REQUIRE(dcResApply == fdOp.apply(dc));

                RealVector_t resApplyAdjoint(fdOp.getDomainDescriptor().getNumberOfCoefficients());
                resApplyAdjoint << 57, -26, 2, -11, 8;
                DataContainer dcResApplyAdjoint(fdOp.getDomainDescriptor(), resApplyAdjoint);

                REQUIRE(dcResApplyAdjoint == fdOp.applyAdjoint(dcResApply));
            }
        }

        WHEN("using central differences")
        {
            FiniteDifferences fdOp(dd, FiniteDifferences<real_t>::DiffType::CENTRAL);

            THEN("the results are correct")
            {
                RealVector_t resApply(fdOp.getRangeDescriptor().getNumberOfCoefficients());
                resApply << 1.5, -14.0, -2.0, 2.5, 0.5;
                DataContainer dcResApply(fdOp.getRangeDescriptor(), resApply);

                REQUIRE(dcResApply == fdOp.apply(dc));

                RealVector_t resApplyAdjoint(fdOp.getDomainDescriptor().getNumberOfCoefficients());
                resApplyAdjoint << 7.0, 1.75, -8.25, -1.25, 1.25;
                DataContainer dcResApplyAdjoint(fdOp.getDomainDescriptor(), resApplyAdjoint);

                REQUIRE(dcResApplyAdjoint == fdOp.applyAdjoint(dcResApply));
            }
        }
    }
}

SCENARIO("Testing FiniteDifferences in 2D")
{
    GIVEN("some data")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 4, 4;
        VolumeDescriptor dd(numCoeff);
        RealVector_t data(dd.getNumberOfCoefficients());
        data << 16, 5, 9, 4, 2, 11, 7, 14, 3, 10, 6, 15, 13, 8, 12, 1;
        DataContainer dc(dd, data);

        WHEN("using forward differences (default)")
        {
            FiniteDifferences fdOp(dd);

            THEN("the results are correct")
            {
                RealVector_t res(fdOp.getRangeDescriptor().getNumberOfCoefficients());
                res << -11, 4, -5, -4, 9, -4, 7, -14, 7, -4, 9, -15, -5, 4, -11, -1, -14, 6, -2, 10,
                    1, -1, -1, 1, 10, -2, 6, -14, -13, -8, -12, -1;
                DataContainer dcRes(fdOp.getRangeDescriptor(), res);

                REQUIRE(dcRes == fdOp.apply(dc));
            }
        }

        WHEN("using backward differences")
        {
            FiniteDifferences fdOp(dd, FiniteDifferences<real_t>::DiffType::BACKWARD);

            THEN("the results are correct")
            {
                RealVector_t res(fdOp.getRangeDescriptor().getNumberOfCoefficients());
                res << 16, -11, 4, -5, 2, 9, -4, 7, 3, 7, -4, 9, 13, -5, 4, -11, 16, 5, 9, 4, -14,
                    6, -2, 10, 1, -1, -1, 1, 10, -2, 6, -14;
                DataContainer dcRes(fdOp.getRangeDescriptor(), res);

                REQUIRE(dcRes == fdOp.apply(dc));
            }
        }

        WHEN("using central differences")
        {
            FiniteDifferences fdOp(dd, FiniteDifferences<real_t>::DiffType::CENTRAL);

            THEN("the results are correct")
            {
                RealVector_t res(fdOp.getRangeDescriptor().getNumberOfCoefficients());
                res << 2.5, -3.5, -0.5, -4.5, 5.5, 2.5, 1.5, -3.5, 5.0, 1.5, 2.5, -3.0, 4.0, -0.5,
                    -3.5, -6.0,
                    //
                    1.0, 5.5, 3.5, 7.0, -6.5, 2.5, -1.5, 5.5, 5.5, -1.5, 2.5, -6.5, -1.5, -5.0,
                    -3.0, -7.5;
                DataContainer dcRes(fdOp.getRangeDescriptor(), res);

                REQUIRE(dcRes == fdOp.apply(dc));
            }
        }
    }
}

SCENARIO("Testing FiniteDifferences in 2D with not all dimensions active")
{
    GIVEN("some data")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 4, 4;
        VolumeDescriptor dd(numCoeff);
        RealVector_t data(dd.getNumberOfCoefficients());
        data << 16, 5, 9, 4, 2, 11, 7, 14, 3, 10, 6, 15, 13, 8, 12, 1;
        DataContainer dc(dd, data);

        WHEN("using forward differences (default)")
        {
            BooleanVector_t activeDims(2);
            activeDims << true, false;
            FiniteDifferences fdOp1(dd, activeDims);

            activeDims << false, true;
            FiniteDifferences fdOp2(dd, activeDims);

            THEN("the results are correct")
            {
                RealVector_t res1(fdOp1.getRangeDescriptor().getNumberOfCoefficients());
                res1 << -11, 4, -5, -4, 9, -4, 7, -14, 7, -4, 9, -15, -5, 4, -11, -1;
                DataContainer dcRes1(fdOp1.getRangeDescriptor(), res1);

                REQUIRE(dcRes1 == fdOp1.apply(dc));

                RealVector_t res2(fdOp2.getRangeDescriptor().getNumberOfCoefficients());
                res2 << -14, 6, -2, 10, 1, -1, -1, 1, 10, -2, 6, -14, -13, -8, -12, -1;
                DataContainer dcRes2(fdOp2.getRangeDescriptor(), res2);

                REQUIRE(dcRes2 == fdOp2.apply(dc));
            }
        }

        WHEN("using backward differences")
        {
            BooleanVector_t activeDims(2);
            activeDims << true, false;
            FiniteDifferences fdOp1(dd, activeDims, FiniteDifferences<real_t>::DiffType::BACKWARD);

            activeDims << false, true;
            FiniteDifferences fdOp2(dd, activeDims, FiniteDifferences<real_t>::DiffType::BACKWARD);

            THEN("the results are correct")
            {
                RealVector_t res1(fdOp1.getRangeDescriptor().getNumberOfCoefficients());
                res1 << 16, -11, 4, -5, 2, 9, -4, 7, 3, 7, -4, 9, 13, -5, 4, -11;
                DataContainer dcRes1(fdOp1.getRangeDescriptor(), res1);

                REQUIRE(dcRes1 == fdOp1.apply(dc));

                RealVector_t res2(fdOp2.getRangeDescriptor().getNumberOfCoefficients());
                res2 << 16, 5, 9, 4, -14, 6, -2, 10, 1, -1, -1, 1, 10, -2, 6, -14;
                DataContainer dcRes2(fdOp2.getRangeDescriptor(), res2);

                REQUIRE(dcRes2 == fdOp2.apply(dc));
            }
        }

        WHEN("using central differences")
        {
            BooleanVector_t activeDims(2);
            activeDims << true, false;
            FiniteDifferences fdOp1(dd, activeDims, FiniteDifferences<real_t>::DiffType::CENTRAL);

            activeDims << false, true;
            FiniteDifferences fdOp2(dd, activeDims, FiniteDifferences<real_t>::DiffType::CENTRAL);

            THEN("the results are correct")
            {
                RealVector_t res1(fdOp1.getRangeDescriptor().getNumberOfCoefficients());
                res1 << 2.5, -3.5, -0.5, -4.5, 5.5, 2.5, 1.5, -3.5, 5.0, 1.5, 2.5, -3.0, 4.0, -0.5,
                    -3.5, -6.0;
                DataContainer dcRes1(fdOp1.getRangeDescriptor(), res1);

                REQUIRE(dcRes1 == fdOp1.apply(dc));

                RealVector_t res2(fdOp2.getRangeDescriptor().getNumberOfCoefficients());
                res2 << 1.0, 5.5, 3.5, 7.0, -6.5, 2.5, -1.5, 5.5, 5.5, -1.5, 2.5, -6.5, -1.5, -5.0,
                    -3.0, -7.5;
                DataContainer dcRes2(fdOp2.getRangeDescriptor(), res2);

                REQUIRE(dcRes2 == fdOp2.apply(dc));
            }
        }
    }
}
