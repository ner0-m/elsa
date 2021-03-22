/**
 * @file test_SoftThresholding.cpp
 *
 * @brief Tests for the SoftThresholding class
 *
 * @author Andi Braimllari
 */

#include "SoftThresholding.h"
#include "VolumeDescriptor.h"

#include <catch2/catch.hpp>
#include <testHelpers.h>

using namespace elsa;

SCENARIO("Constructing SoftThresholding")
{
    GIVEN("a DataDescriptor")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 45, 11, 7;
        VolumeDescriptor volDescr(numCoeff);

        WHEN("instantiating a SoftThresholding operator")
        {
            SoftThresholding<real_t> sThrOp(volDescr);

            THEN("the DataDescriptors are equal")
            {
                REQUIRE(sThrOp.getRangeDescriptor() == volDescr);
            }
        }

        WHEN("cloning a SoftThresholding operator")
        {
            SoftThresholding<real_t> sThrOp(volDescr);
            auto sThrOpClone = sThrOp.clone();

            THEN("cloned SoftThresholding operator equals original SoftThresholding operator")
            {
                REQUIRE(sThrOpClone.get() != &sThrOp);
                REQUIRE(*sThrOpClone == sThrOp);
            }
        }
    }
}

SCENARIO("Using SoftThresholding in 1D")
{
    GIVEN("a DataDescriptor")
    {
        IndexVector_t numCoeff(1);
        numCoeff << 8;
        VolumeDescriptor volDescr(numCoeff);

        WHEN("Using SoftThresholding operator in 1D")
        {
            SoftThresholding<real_t> sThrOp(volDescr);

            THEN("Values under threshold=4 are 0 and values above are sign(v) * (abs(v) - t)")
            {
                RealVector_t data(volDescr.getNumberOfCoefficients());
                data << -2, 3, 4, -7, 7, 8, 8, 3;
                DataContainer<real_t> dataCont(volDescr, data);

                RealVector_t expectedRes(sThrOp.getRangeDescriptor().getNumberOfCoefficients());
                expectedRes << 0, 0, 0, -3, 3, 4, 4, 0;
                DataContainer<real_t> dCRes(sThrOp.getRangeDescriptor(), expectedRes);

                REQUIRE(isApprox(dCRes, sThrOp.apply(dataCont, geometry::Threshold<real_t>{4})));
            }
        }
    }
}

SCENARIO("Using SoftThresholding in 3D")
{
    GIVEN("a DataDescriptor")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 3, 2, 3;
        VolumeDescriptor volDescr(numCoeff);

        WHEN("Using SoftThresholding operator in 3D")
        {
            SoftThresholding<real_t> sThrOp(volDescr);

            THEN("Values under threshold=5 are 0 and values above are sign(v) * (abs(v) - t)")
            {
                RealVector_t data(volDescr.getNumberOfCoefficients());
                data << 2, 1, 6, 6, 1, 4, 2, -9, 7, 7, 7, 3, 1, 2, 8, 9, -4, 5;
                DataContainer<real_t> dataCont(volDescr, data);

                RealVector_t expectedRes(sThrOp.getRangeDescriptor().getNumberOfCoefficients());
                expectedRes << 0, 0, 1, 1, 0, 0, 0, -4, 2, 2, 2, 0, 0, 0, 3, 4, 0, 0;
                DataContainer<real_t> dCRes(sThrOp.getRangeDescriptor(), expectedRes);

                REQUIRE(isApprox(dCRes, sThrOp.apply(dataCont, geometry::Threshold<real_t>{5})));
            }
        }
    }
}

SCENARIO("Using SoftThresholding")
{
    GIVEN("a DataDescriptor")
    {
        IndexVector_t numCoeff(1);
        numCoeff << 8;
        VolumeDescriptor volDescr(numCoeff);

        WHEN("Using SoftThresholding operator")
        {
            SoftThresholding<real_t> sThrOp(volDescr);

            THEN("The zero vector is returned when the zero vector is given")
            {
                RealVector_t data(volDescr.getNumberOfCoefficients());
                data << 0, 0, 0, 0, 0, 0, 0, 0;
                DataContainer<real_t> dataCont(volDescr, data);

                RealVector_t expectedRes(sThrOp.getRangeDescriptor().getNumberOfCoefficients());
                expectedRes << 0, 0, 0, 0, 0, 0, 0, 0;
                DataContainer<real_t> dCRes(sThrOp.getRangeDescriptor(), expectedRes);

                REQUIRE(isApprox(dCRes, sThrOp.apply(dataCont, geometry::Threshold<real_t>{4})));
            }

            THEN("SoftThresholding operator throws exception for t = 0")
            {
                RealVector_t data(volDescr.getNumberOfCoefficients());
                data << 0, 0, 0, 0, 0, 0, 0, 0;
                DataContainer<real_t> dC(volDescr, data);

                REQUIRE_THROWS_AS(sThrOp.apply(dC, geometry::Threshold<real_t>{0}),
                                  std::invalid_argument);
            }

            THEN("SoftThresholding operator throws exception for t < 0")
            {
                RealVector_t data(volDescr.getNumberOfCoefficients());
                data << 0, 0, 0, 0, 0, 0, 0, 0;
                DataContainer<real_t> dataCont(volDescr, data);

                REQUIRE_THROWS_AS(sThrOp.apply(dataCont, geometry::Threshold<real_t>{-1}),
                                  std::invalid_argument);
            }

            THEN("SoftThresholding operator throws exception for differently sized v and prox")
            {
                RealVector_t data(volDescr.getNumberOfCoefficients());
                data << 0, 0, 0, 0, 0, 0, 0, 0;
                DataContainer<real_t> dC(volDescr, data);

                IndexVector_t numCoeff1(1);
                numCoeff1 << 9;
                VolumeDescriptor volDescr1(numCoeff1);
                RealVector_t data1(volDescr1.getNumberOfCoefficients());
                data1 << 0, 0, 0, 0, 0, 0, 0, 0, 0;
                DataContainer<real_t> dC1(volDescr1, data1);

                REQUIRE_THROWS_AS(sThrOp.apply(dC, geometry::Threshold<real_t>{-1}, dC1),
                                  std::logic_error);
            }
        }
    }
}
