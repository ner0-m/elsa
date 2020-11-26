#include "HardThresholding.h"
#include "VolumeDescriptor.h"

#include <catch2/catch.hpp>
#include <testHelpers.h>

using namespace elsa;

SCENARIO("Constructing HardThresholding")
{
    GIVEN("a DataDescriptor")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 8, 4, 52;
        VolumeDescriptor volDescr(numCoeff);

        WHEN("instantiating a HardThresholding operator")
        {
            HardThresholding<real_t> hThrOp(volDescr);

            THEN("the DataDescriptors are equal")
            {
                REQUIRE(hThrOp.getRangeDescriptor() == volDescr);
            }
        }

        WHEN("cloning a HardThresholding operator")
        {
            HardThresholding<real_t> hThrOp(volDescr);
            auto hThrOpClone = hThrOp.clone();

            THEN("cloned HardThresholding operator equals original HardThresholding operator")
            {
                REQUIRE(hThrOpClone.get() != &hThrOp);
                REQUIRE(*hThrOpClone == hThrOp);
            }
        }
    }
}

SCENARIO("Using HardThresholding in 1D")
{
    GIVEN("a DataDescriptor")
    {
        IndexVector_t numCoeff(1);
        numCoeff << 8;
        VolumeDescriptor volDescr(numCoeff);

        WHEN("Using HardThresholding operator in 1D")
        {
            HardThresholding<real_t> hThrOp(volDescr);

            THEN("Values under threshold=4 are 0 and values above remain the same")
            {
                RealVector_t data(volDescr.getNumberOfCoefficients());
                data << -2, 3, 4, -7, 7, 8, 8, 3;
                DataContainer<real_t> dC(volDescr, data);

                RealVector_t expectedRes(hThrOp.getRangeDescriptor().getNumberOfCoefficients());
                expectedRes << 0, 0, 0, -7, 7, 8, 8, 0;
                DataContainer<real_t> dCRes(hThrOp.getRangeDescriptor(), expectedRes);

                REQUIRE(isApprox(dCRes, hThrOp.apply(dC, geometry::Threshold<real_t>{4})));
            }
        }
    }
}

SCENARIO("Using HardThresholding in 3D")
{
    GIVEN("a DataDescriptor")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 3, 2, 3;
        VolumeDescriptor volumeDescriptor(numCoeff);

        WHEN("Using HardThresholding operator in 3D")
        {
            HardThresholding<real_t> hThrOp(volumeDescriptor);

            THEN("Values under threshold=5 are 0 and values above remain the same")
            {
                RealVector_t data(volumeDescriptor.getNumberOfCoefficients());
                data << 2, 1, 6, 6, 1, 4, 2, -9, 7, 7, 7, 3, 1, 2, 8, 9, -4, 5;
                DataContainer<real_t> dC(volumeDescriptor, data);

                RealVector_t expectedRes(hThrOp.getRangeDescriptor().getNumberOfCoefficients());
                expectedRes << 0, 0, 6, 6, 0, 0, 0, -9, 7, 7, 7, 0, 0, 0, 8, 9, 0, 0;
                DataContainer<real_t> dCRes(hThrOp.getRangeDescriptor(), expectedRes);

                REQUIRE(isApprox(dCRes, hThrOp.apply(dC, geometry::Threshold<real_t>{5})));
            }
        }
    }
}

SCENARIO("Using HardThresholding")
{
    GIVEN("a DataDescriptor")
    {
        IndexVector_t numCoeff(1);
        numCoeff << 8;
        VolumeDescriptor volDescr(numCoeff);

        WHEN("Using HardThresholding operator")
        {
            HardThresholding<real_t> hThrOp(volDescr);

            THEN("The zero vector is returned when the zero vector is given")
            {
                RealVector_t data(volDescr.getNumberOfCoefficients());
                data << 0, 0, 0, 0, 0, 0, 0, 0;
                DataContainer<real_t> dataContainer(volDescr, data);

                RealVector_t expectedRes(hThrOp.getRangeDescriptor().getNumberOfCoefficients());
                expectedRes << 0, 0, 0, 0, 0, 0, 0, 0;
                DataContainer<real_t> dCRes(hThrOp.getRangeDescriptor(), expectedRes);

                REQUIRE(
                    isApprox(dCRes, hThrOp.apply(dataContainer, geometry::Threshold<real_t>{4})));
            }

            THEN("HardThresholding operator throws exception for t = 0")
            {
                RealVector_t data(volDescr.getNumberOfCoefficients());
                data << 0, 0, 0, 0, 0, 0, 0, 0;
                DataContainer<real_t> dC(volDescr, data);

                REQUIRE_THROWS_AS(hThrOp.apply(dC, geometry::Threshold<real_t>{0}),
                                  std::invalid_argument);
            }

            THEN("HardThresholding operator throws exception for t < 0")
            {
                RealVector_t data(volDescr.getNumberOfCoefficients());
                data << 0, 0, 0, 0, 0, 0, 0, 0;
                DataContainer<real_t> dC(volDescr, data);

                REQUIRE_THROWS_AS(hThrOp.apply(dC, geometry::Threshold<real_t>{-1}),
                                  std::invalid_argument);
            }

            THEN("HardThresholding operator throws exception for differently sized v and prox")
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

                REQUIRE_THROWS_AS(hThrOp.apply(dC, geometry::Threshold<real_t>{-1}, dC1),
                                  std::logic_error);
            }
        }
    }
}
