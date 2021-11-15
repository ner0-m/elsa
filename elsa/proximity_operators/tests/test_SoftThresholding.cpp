/**
 * @file test_SoftThresholding.cpp
 *
 * @brief Tests for the SoftThresholding class
 *
 * @author Andi Braimllari
 */

#include "Error.h"
#include "SoftThresholding.h"
#include "VolumeDescriptor.h"

#include "doctest/doctest.h"
#include <testHelpers.h>

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("proximity_operators");

TEST_CASE_TEMPLATE("SoftThresholding: Testing construction", data_t, float, double)
{
    GIVEN("a DataDescriptor")
    {
        VolumeDescriptor volDescr({45, 11, 7});

        WHEN("instantiating a SoftThresholding operator")
        {
            SoftThresholding<data_t> sThrOp(volDescr);

            THEN("the DataDescriptors are equal")
            {
                REQUIRE_EQ(sThrOp.getRangeDescriptor(), volDescr);
            }
        }

        WHEN("cloning a SoftThresholding operator")
        {
            SoftThresholding<data_t> sThrOp(volDescr);
            auto sThrOpClone = sThrOp.clone();

            THEN("cloned SoftThresholding operator equals original SoftThresholding operator")
            {
                REQUIRE_NE(sThrOpClone.get(), &sThrOp);
                REQUIRE_EQ(*sThrOpClone, sThrOp);
            }
        }
    }
}

TEST_CASE_TEMPLATE("SoftThresholding: Testing in 1D", data_t, float, double)
{
    GIVEN("a DataDescriptor")
    {
        VolumeDescriptor volDescr({8});

        WHEN("Using SoftThresholding operator in 1D")
        {
            SoftThresholding<data_t> sThrOp(volDescr);

            THEN("Values under threshold=4 are 0 and values above are sign(v) * (abs(v) - t)")
            {
                Vector_t<data_t> data(volDescr.getNumberOfCoefficients());
                data << -2, 3, 4, -7, 7, 8, 8, 3;
                DataContainer<data_t> dataCont(volDescr, data);

                Vector_t<data_t> expectedRes(sThrOp.getRangeDescriptor().getNumberOfCoefficients());
                expectedRes << 0, 0, 0, -3, 3, 4, 4, 0;
                DataContainer<data_t> dCRes(sThrOp.getRangeDescriptor(), expectedRes);

                REQUIRE_UNARY(
                    isApprox(dCRes, sThrOp.apply(dataCont, geometry::Threshold<data_t>{4})));
            }
        }
    }
}

TEST_CASE_TEMPLATE("SoftThresholding: Testing in 3D", data_t, float, double)
{
    GIVEN("a DataDescriptor")
    {
        VolumeDescriptor volDescr({3, 2, 3});

        WHEN("Using SoftThresholding operator in 3D")
        {
            SoftThresholding<data_t> sThrOp(volDescr);

            THEN("Values under threshold=5 are 0 and values above are sign(v) * (abs(v) - t)")
            {
                Vector_t<data_t> data(volDescr.getNumberOfCoefficients());
                data << 2, 1, 6, 6, 1, 4, 2, -9, 7, 7, 7, 3, 1, 2, 8, 9, -4, 5;
                DataContainer<data_t> dataCont(volDescr, data);

                Vector_t<data_t> expectedRes(sThrOp.getRangeDescriptor().getNumberOfCoefficients());
                expectedRes << 0, 0, 1, 1, 0, 0, 0, -4, 2, 2, 2, 0, 0, 0, 3, 4, 0, 0;
                DataContainer<data_t> dCRes(sThrOp.getRangeDescriptor(), expectedRes);

                REQUIRE_UNARY(
                    isApprox(dCRes, sThrOp.apply(dataCont, geometry::Threshold<data_t>{5})));
            }
        }
    }
}

TEST_CASE_TEMPLATE("SoftThresholding: Testing general behaviour", data_t, float, double)
{
    GIVEN("a DataDescriptor")
    {
        VolumeDescriptor volDescr({8});

        WHEN("Using SoftThresholding operator")
        {
            SoftThresholding<data_t> sThrOp(volDescr);

            THEN("The zero vector is returned when the zero vector is given")
            {
                Vector_t<data_t> data(volDescr.getNumberOfCoefficients());
                data << 0, 0, 0, 0, 0, 0, 0, 0;
                DataContainer<data_t> dataCont(volDescr, data);

                Vector_t<data_t> expectedRes(sThrOp.getRangeDescriptor().getNumberOfCoefficients());
                expectedRes << 0, 0, 0, 0, 0, 0, 0, 0;
                DataContainer<data_t> dCRes(sThrOp.getRangeDescriptor(), expectedRes);

                REQUIRE_UNARY(
                    isApprox(dCRes, sThrOp.apply(dataCont, geometry::Threshold<data_t>{4})));
            }

            THEN("SoftThresholding operator throws exception for t = 0")
            {
                Vector_t<data_t> data(volDescr.getNumberOfCoefficients());
                data << 0, 0, 0, 0, 0, 0, 0, 0;
                DataContainer<data_t> dC(volDescr, data);

                // actually the geometry::Threshold throws this
                REQUIRE_THROWS_AS(sThrOp.apply(dC, geometry::Threshold<data_t>{0}),
                                  InvalidArgumentError);
            }

            THEN("SoftThresholding operator throws exception for t < 0")
            {
                Vector_t<data_t> data(volDescr.getNumberOfCoefficients());
                data << 0, 0, 0, 0, 0, 0, 0, 0;
                DataContainer<data_t> dataCont(volDescr, data);

                // actually the geometry::Threshold throws this
                REQUIRE_THROWS_AS(sThrOp.apply(dataCont, geometry::Threshold<data_t>{-1}),
                                  InvalidArgumentError);
            }

            THEN("SoftThresholding operator throws exception for differently sized v and prox")
            {
                Vector_t<data_t> data(volDescr.getNumberOfCoefficients());
                data << 0, 0, 0, 0, 0, 0, 0, 0;
                DataContainer<data_t> dC(volDescr, data);

                VolumeDescriptor volDescr1({9});
                Vector_t<data_t> data1(volDescr1.getNumberOfCoefficients());
                data1 << 0, 0, 0, 0, 0, 0, 0, 0, 0;
                DataContainer<data_t> dC1(volDescr1, data1);

                REQUIRE_THROWS_AS(sThrOp.apply(dC, geometry::Threshold<data_t>{1}, dC1),
                                  LogicError);
            }
        }
    }
}

TEST_SUITE_END();
