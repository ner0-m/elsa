/**
 * @file test_FiniteDifferences.cpp
 *
 * @brief Tests for FiniteDifferences class
 *
 * @author Matthias Wieczorek - main code
 * @author Tobias Lasser - rewrite
 */

#include "doctest/doctest.h"
#include "FiniteDifferences.h"
#include "VolumeDescriptor.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("core");

TEST_CASE_TEMPLATE("FiniteDifference: Testing construction", data_t, float, double)
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
            FiniteDifferences<data_t> fdOp(dd);

            THEN("the descriptors are as expected")
            {
                REQUIRE_EQ(fdOp.getDomainDescriptor(), dd);
                REQUIRE_EQ(fdOp.getRangeDescriptor(), ddRange);
            }
        }

        WHEN("cloning a FiniteDifference operator")
        {
            FiniteDifferences<data_t> fdOp(dd);
            auto fdOpClone = fdOp.clone();

            THEN("everything matches")
            {
                REQUIRE_NE(fdOpClone.get(), &fdOp);
                REQUIRE_EQ(*fdOpClone, fdOp);
            }
        }
    }
}

TEST_CASE_TEMPLATE("FiniteDifference: Testing in 1D", data_t, float, double)
{
    GIVEN("some data")
    {
        IndexVector_t numCoeff(1);
        numCoeff << 5;
        VolumeDescriptor dd(numCoeff);
        Vector_t<data_t> data(dd.getNumberOfCoefficients());
        data << 30, 3, 2, -1, 7;

        DataContainer<data_t> dc(dd, data);

        WHEN("using forward differences (default mode)")
        {
            FiniteDifferences<data_t> fdOp(dd);

            THEN("the results are correct")
            {
                Vector_t<data_t> resApply(fdOp.getRangeDescriptor().getNumberOfCoefficients());
                resApply << -27, -1, -3, 8, -7;
                DataContainer<data_t> dcResApply(fdOp.getRangeDescriptor(), resApply);

                REQUIRE_EQ(dcResApply, fdOp.apply(dc));

                Vector_t<data_t> resApplyAdjoint(
                    fdOp.getDomainDescriptor().getNumberOfCoefficients());
                resApplyAdjoint << 27, -26, 2, -11, 15;
                DataContainer<data_t> dcResApplyAdjoint(fdOp.getDomainDescriptor(),
                                                        resApplyAdjoint);

                REQUIRE_EQ(dcResApplyAdjoint, fdOp.applyAdjoint(dcResApply));
            }
        }

        WHEN("using backward differences")
        {
            FiniteDifferences<data_t> fdOp(dd, FiniteDifferences<data_t>::DiffType::BACKWARD);

            THEN("the results are correct")
            {
                Vector_t<data_t> resApply(fdOp.getRangeDescriptor().getNumberOfCoefficients());
                resApply << 30, -27, -1, -3, 8;
                DataContainer<data_t> dcResApply(fdOp.getRangeDescriptor(), resApply);

                REQUIRE_EQ(dcResApply, fdOp.apply(dc));

                Vector_t<data_t> resApplyAdjoint(
                    fdOp.getDomainDescriptor().getNumberOfCoefficients());
                resApplyAdjoint << 57, -26, 2, -11, 8;
                DataContainer<data_t> dcResApplyAdjoint(fdOp.getDomainDescriptor(),
                                                        resApplyAdjoint);

                REQUIRE_EQ(dcResApplyAdjoint, fdOp.applyAdjoint(dcResApply));
            }
        }

        WHEN("using central differences")
        {
            FiniteDifferences<data_t> fdOp(dd, FiniteDifferences<data_t>::DiffType::CENTRAL);

            THEN("the results are correct")
            {
                Vector_t<data_t> resApply(fdOp.getRangeDescriptor().getNumberOfCoefficients());
                resApply << 1.5, -14.0, -2.0, 2.5, 0.5;
                DataContainer<data_t> dcResApply(fdOp.getRangeDescriptor(), resApply);

                REQUIRE_EQ(dcResApply, fdOp.apply(dc));

                Vector_t<data_t> resApplyAdjoint(
                    fdOp.getDomainDescriptor().getNumberOfCoefficients());
                resApplyAdjoint << 7.0, 1.75, -8.25, -1.25, 1.25;
                DataContainer<data_t> dcResApplyAdjoint(fdOp.getDomainDescriptor(),
                                                        resApplyAdjoint);

                REQUIRE_EQ(dcResApplyAdjoint, fdOp.applyAdjoint(dcResApply));
            }
        }
    }
}

TEST_CASE_TEMPLATE("FiniteDifference: Testing in 2D", data_t, float, double)
{
    GIVEN("some data")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 4, 4;
        VolumeDescriptor dd(numCoeff);
        Vector_t<data_t> data(dd.getNumberOfCoefficients());
        data << 16, 5, 9, 4, 2, 11, 7, 14, 3, 10, 6, 15, 13, 8, 12, 1;
        DataContainer<data_t> dc(dd, data);

        WHEN("using forward differences (default)")
        {
            FiniteDifferences<data_t> fdOp(dd);

            THEN("the results are correct")
            {
                Vector_t<data_t> res(fdOp.getRangeDescriptor().getNumberOfCoefficients());
                res << -11, 4, -5, -4, 9, -4, 7, -14, 7, -4, 9, -15, -5, 4, -11, -1, -14, 6, -2, 10,
                    1, -1, -1, 1, 10, -2, 6, -14, -13, -8, -12, -1;
                DataContainer<data_t> dcRes(fdOp.getRangeDescriptor(), res);

                REQUIRE_EQ(dcRes, fdOp.apply(dc));
            }
        }

        WHEN("using backward differences")
        {
            FiniteDifferences<data_t> fdOp(dd, FiniteDifferences<data_t>::DiffType::BACKWARD);

            THEN("the results are correct")
            {
                Vector_t<data_t> res(fdOp.getRangeDescriptor().getNumberOfCoefficients());
                res << 16, -11, 4, -5, 2, 9, -4, 7, 3, 7, -4, 9, 13, -5, 4, -11, 16, 5, 9, 4, -14,
                    6, -2, 10, 1, -1, -1, 1, 10, -2, 6, -14;
                DataContainer<data_t> dcRes(fdOp.getRangeDescriptor(), res);

                REQUIRE_EQ(dcRes, fdOp.apply(dc));
            }
        }

        WHEN("using central differences")
        {
            FiniteDifferences<data_t> fdOp(dd, FiniteDifferences<data_t>::DiffType::CENTRAL);

            THEN("the results are correct")
            {
                Vector_t<data_t> res(fdOp.getRangeDescriptor().getNumberOfCoefficients());
                res << 2.5, -3.5, -0.5, -4.5, 5.5, 2.5, 1.5, -3.5, 5.0, 1.5, 2.5, -3.0, 4.0, -0.5,
                    -3.5, -6.0,
                    //
                    1.0, 5.5, 3.5, 7.0, -6.5, 2.5, -1.5, 5.5, 5.5, -1.5, 2.5, -6.5, -1.5, -5.0,
                    -3.0, -7.5;
                DataContainer<data_t> dcRes(fdOp.getRangeDescriptor(), res);

                REQUIRE_EQ(dcRes, fdOp.apply(dc));
            }
        }
    }
}

TEST_CASE_TEMPLATE("FiniteDifference: Testing in 2D with not all dimensions active", data_t, float,
                   double)
{
    GIVEN("some data")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 4, 4;
        VolumeDescriptor dd(numCoeff);
        Vector_t<data_t> data(dd.getNumberOfCoefficients());
        data << 16, 5, 9, 4, 2, 11, 7, 14, 3, 10, 6, 15, 13, 8, 12, 1;
        DataContainer<data_t> dc(dd, data);

        WHEN("using forward differences (default)")
        {
            BooleanVector_t activeDims(2);
            activeDims << true, false;
            FiniteDifferences<data_t> fdOp1(dd, activeDims);

            activeDims << false, true;
            FiniteDifferences<data_t> fdOp2(dd, activeDims);

            THEN("the results are correct")
            {
                Vector_t<data_t> res1(fdOp1.getRangeDescriptor().getNumberOfCoefficients());
                res1 << -11, 4, -5, -4, 9, -4, 7, -14, 7, -4, 9, -15, -5, 4, -11, -1;
                DataContainer<data_t> dcRes1(fdOp1.getRangeDescriptor(), res1);

                REQUIRE_EQ(dcRes1, fdOp1.apply(dc));

                Vector_t<data_t> res2(fdOp2.getRangeDescriptor().getNumberOfCoefficients());
                res2 << -14, 6, -2, 10, 1, -1, -1, 1, 10, -2, 6, -14, -13, -8, -12, -1;
                DataContainer<data_t> dcRes2(fdOp2.getRangeDescriptor(), res2);

                REQUIRE_EQ(dcRes2, fdOp2.apply(dc));
            }
        }

        WHEN("using backward differences")
        {
            BooleanVector_t activeDims(2);
            activeDims << true, false;
            FiniteDifferences<data_t> fdOp1(dd, activeDims,
                                            FiniteDifferences<data_t>::DiffType::BACKWARD);

            activeDims << false, true;
            FiniteDifferences<data_t> fdOp2(dd, activeDims,
                                            FiniteDifferences<data_t>::DiffType::BACKWARD);

            THEN("the results are correct")
            {
                Vector_t<data_t> res1(fdOp1.getRangeDescriptor().getNumberOfCoefficients());
                res1 << 16, -11, 4, -5, 2, 9, -4, 7, 3, 7, -4, 9, 13, -5, 4, -11;
                DataContainer<data_t> dcRes1(fdOp1.getRangeDescriptor(), res1);

                REQUIRE_EQ(dcRes1, fdOp1.apply(dc));

                Vector_t<data_t> res2(fdOp2.getRangeDescriptor().getNumberOfCoefficients());
                res2 << 16, 5, 9, 4, -14, 6, -2, 10, 1, -1, -1, 1, 10, -2, 6, -14;
                DataContainer<data_t> dcRes2(fdOp2.getRangeDescriptor(), res2);

                REQUIRE_EQ(dcRes2, fdOp2.apply(dc));
            }
        }

        WHEN("using central differences")
        {
            BooleanVector_t activeDims(2);
            activeDims << true, false;
            FiniteDifferences<data_t> fdOp1(dd, activeDims,
                                            FiniteDifferences<data_t>::DiffType::CENTRAL);

            activeDims << false, true;
            FiniteDifferences<data_t> fdOp2(dd, activeDims,
                                            FiniteDifferences<data_t>::DiffType::CENTRAL);

            THEN("the results are correct")
            {
                Vector_t<data_t> res1(fdOp1.getRangeDescriptor().getNumberOfCoefficients());
                res1 << 2.5, -3.5, -0.5, -4.5, 5.5, 2.5, 1.5, -3.5, 5.0, 1.5, 2.5, -3.0, 4.0, -0.5,
                    -3.5, -6.0;
                DataContainer<data_t> dcRes1(fdOp1.getRangeDescriptor(), res1);

                REQUIRE_EQ(dcRes1, fdOp1.apply(dc));

                Vector_t<data_t> res2(fdOp2.getRangeDescriptor().getNumberOfCoefficients());
                res2 << 1.0, 5.5, 3.5, 7.0, -6.5, 2.5, -1.5, 5.5, 5.5, -1.5, 2.5, -6.5, -1.5, -5.0,
                    -3.0, -7.5;
                DataContainer<data_t> dcRes2(fdOp2.getRangeDescriptor(), res2);

                REQUIRE_EQ(dcRes2, fdOp2.apply(dc));
            }
        }
    }
}
TEST_SUITE_END();
