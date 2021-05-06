/**
 * @file test_HardThresholding.cpp
 *
 * @brief Tests for the HardThresholding class
 *
 * @author Andi Braimllari
 */

#include "Error.h"
#include "HardThresholding.h"
#include "VolumeDescriptor.h"

#include "doctest/doctest.h"
#include <testHelpers.h>

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("proximity_operators");

TEST_CASE_TEMPLATE("HardThresholding: Testing construction", data_t, float, double)
{
    GIVEN("a DataDescriptor")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 8, 4, 52;
        VolumeDescriptor volDescr(numCoeff);

        WHEN("instantiating a HardThresholding operator")
        {
            HardThresholding<data_t> hThrOp(volDescr);

            THEN("the DataDescriptors are equal")
            {
                REQUIRE(hThrOp.getRangeDescriptor() == volDescr);
            }
        }

        WHEN("cloning a HardThresholding operator")
        {
            HardThresholding<data_t> hThrOp(volDescr);
            auto hThrOpClone = hThrOp.clone();

            THEN("cloned HardThresholding operator equals original HardThresholding operator")
            {
                REQUIRE(hThrOpClone.get() != &hThrOp);
                REQUIRE(*hThrOpClone == hThrOp);
            }
        }
    }
}

TEST_CASE_TEMPLATE("HardThresholding: Testing in 1D", data_t, float, double)
{
    GIVEN("a DataDescriptor")
    {
        IndexVector_t numCoeff(1);
        numCoeff << 8;
        VolumeDescriptor volDescr(numCoeff);

        WHEN("Using HardThresholding operator in 1D")
        {
            HardThresholding<data_t> hThrOp(volDescr);

            THEN("Values under threshold=4 are 0 and values above remain the same")
            {
                Vector_t<data_t> data(volDescr.getNumberOfCoefficients());
                data << -2, 3, 4, -7, 7, 8, 8, 3;
                DataContainer<data_t> dC(volDescr, data);

                Vector_t<data_t> expectedRes(hThrOp.getRangeDescriptor().getNumberOfCoefficients());
                expectedRes << 0, 0, 0, -7, 7, 8, 8, 0;
                DataContainer<data_t> dCRes(hThrOp.getRangeDescriptor(), expectedRes);

                REQUIRE(isApprox(dCRes, hThrOp.apply(dC, geometry::Threshold<data_t>{4})));
            }
        }
    }
}

TEST_CASE_TEMPLATE("HardThresholding: Testing in 3D", data_t, float, double)
{
    GIVEN("a DataDescriptor")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 3, 2, 3;
        VolumeDescriptor volumeDescriptor(numCoeff);

        WHEN("Using HardThresholding operator in 3D")
        {
            HardThresholding<data_t> hThrOp(volumeDescriptor);

            THEN("Values under threshold=5 are 0 and values above remain the same")
            {
                Vector_t<data_t> data(volumeDescriptor.getNumberOfCoefficients());
                data << 2, 1, 6, 6, 1, 4, 2, -9, 7, 7, 7, 3, 1, 2, 8, 9, -4, 5;
                DataContainer<data_t> dC(volumeDescriptor, data);

                Vector_t<data_t> expectedRes(hThrOp.getRangeDescriptor().getNumberOfCoefficients());
                expectedRes << 0, 0, 6, 6, 0, 0, 0, -9, 7, 7, 7, 0, 0, 0, 8, 9, 0, 0;
                DataContainer<data_t> dCRes(hThrOp.getRangeDescriptor(), expectedRes);

                REQUIRE(isApprox(dCRes, hThrOp.apply(dC, geometry::Threshold<data_t>{5})));
            }
        }
    }
}

TEST_CASE_TEMPLATE("HardThresholding: Testing general behaviour", data_t, float, double)
{
    GIVEN("a DataDescriptor")
    {
        IndexVector_t numCoeff(1);
        numCoeff << 8;
        VolumeDescriptor volDescr(numCoeff);

        WHEN("Using HardThresholding operator")
        {
            HardThresholding<data_t> hThrOp(volDescr);

            THEN("The zero vector is returned when the zero vector is given")
            {
                Vector_t<data_t> data(volDescr.getNumberOfCoefficients());
                data << 0, 0, 0, 0, 0, 0, 0, 0;
                DataContainer<data_t> dataContainer(volDescr, data);

                Vector_t<data_t> expectedRes(hThrOp.getRangeDescriptor().getNumberOfCoefficients());
                expectedRes << 0, 0, 0, 0, 0, 0, 0, 0;
                DataContainer<data_t> dCRes(hThrOp.getRangeDescriptor(), expectedRes);

                REQUIRE(
                    isApprox(dCRes, hThrOp.apply(dataContainer, geometry::Threshold<data_t>{4})));
            }

            THEN("HardThresholding operator throws exception for t = 0")
            {
                Vector_t<data_t> data(volDescr.getNumberOfCoefficients());
                data << 0, 0, 0, 0, 0, 0, 0, 0;
                DataContainer<data_t> dC(volDescr, data);

                // actually the geometry::Threshold throws this
                REQUIRE_THROWS_AS(hThrOp.apply(dC, geometry::Threshold<data_t>{0}),
                                  InvalidArgumentError);
            }

            THEN("HardThresholding operator throws exception for t < 0")
            {
                Vector_t<data_t> data(volDescr.getNumberOfCoefficients());
                data << 0, 0, 0, 0, 0, 0, 0, 0;
                DataContainer<data_t> dC(volDescr, data);

                // actually the geometry::Threshold throws this
                REQUIRE_THROWS_AS(hThrOp.apply(dC, geometry::Threshold<data_t>{-1}),
                                  InvalidArgumentError);
            }

            THEN("HardThresholding operator throws exception for differently sized v and prox")
            {
                Vector_t<data_t> data(volDescr.getNumberOfCoefficients());
                data << 0, 0, 0, 0, 0, 0, 0, 0;
                DataContainer<data_t> dC(volDescr, data);

                IndexVector_t numCoeff1(1);
                numCoeff1 << 9;
                VolumeDescriptor volDescr1(numCoeff1);
                Vector_t<data_t> data1(volDescr1.getNumberOfCoefficients());
                data1 << 0, 0, 0, 0, 0, 0, 0, 0, 0;
                DataContainer<data_t> dC1(volDescr1, data1);

                REQUIRE_THROWS_AS(hThrOp.apply(dC, geometry::Threshold<data_t>{1}, dC1),
                                  LogicError);
            }
        }
    }
}

TEST_SUITE_END();
