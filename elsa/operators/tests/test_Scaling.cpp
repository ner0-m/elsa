/**
 * \file test_Scaling.cpp
 *
 * \brief Tests for Scaling operator class
 *
 * \author Matthias Wieczorek - initial code
 * \author David Frank - rewrite
 * \author Tobias Lasser - minor extensions
 */

#include <catch2/catch.hpp>
#include "Scaling.h"
#include "VolumeDescriptor.h"

using namespace elsa;

SCENARIO("Constructing a Scaling operator")
{
    GIVEN("a descriptor")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 11, 17;
        VolumeDescriptor dd(numCoeff);

        WHEN("instantiating an isotropic scaling operator")
        {
            real_t scaleFactor = 3.5f;
            Scaling scalingOp(dd, scaleFactor);

            THEN("the descriptors are as expected")
            {
                REQUIRE(scalingOp.getDomainDescriptor() == dd);
                REQUIRE(scalingOp.getRangeDescriptor() == dd);
            }

            THEN("the scaling is isotropic and correct")
            {
                REQUIRE(scalingOp.isIsotropic());
                REQUIRE(scalingOp.getScaleFactor() == scaleFactor);
                REQUIRE_THROWS_AS(scalingOp.getScaleFactors(), std::logic_error);
            }
        }

        WHEN("instantiating an anisotropic scaling operator")
        {
            DataContainer dc(dd);
            dc = 3.5f;
            Scaling scalingOp(dd, dc);

            THEN("the descriptors  are as expected")
            {
                REQUIRE(scalingOp.getDomainDescriptor() == dd);
                REQUIRE(scalingOp.getRangeDescriptor() == dd);
            }

            THEN("the scaling is anisotropic")
            {
                REQUIRE(!scalingOp.isIsotropic());
                REQUIRE(scalingOp.getScaleFactors() == dc);
                REQUIRE_THROWS_AS(scalingOp.getScaleFactor(), std::logic_error);
            }
        }

        WHEN("cloning a scaling operator")
        {
            Scaling scalingOp(dd, 3.5f);
            auto scalingOpClone = scalingOp.clone();

            THEN("everything matches")
            {
                REQUIRE(scalingOpClone.get() != &scalingOp);
                REQUIRE(*scalingOpClone == scalingOp);
            }
        }
    }
}

SCENARIO("Using a Scaling operator")
{
    GIVEN("some data")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 34, 13;
        VolumeDescriptor dd(numCoeff);
        DataContainer input(dd);
        real_t inputScalar{2.5f};
        input = inputScalar;

        WHEN("applying isotropic scaling")
        {
            real_t scaleFactor{3.7f};
            Scaling scalingOp(dd, scaleFactor);

            THEN("apply and applyAdjoint yield the correct results")
            {
                auto resultApply = scalingOp.apply(input);
                for (index_t i = 0; i < resultApply.getSize(); ++i)
                    REQUIRE(resultApply[i] == Approx(inputScalar * scaleFactor));

                auto resultApplyAdjoint = scalingOp.applyAdjoint(input);
                for (index_t i = 0; i < resultApply.getSize(); ++i)
                    REQUIRE(resultApplyAdjoint[i] == Approx(inputScalar * scaleFactor));
            }
        }

        WHEN("applying anisotropic scaling")
        {
            RealVector_t randomData(dd.getNumberOfCoefficients());
            randomData.setRandom();
            DataContainer scaleFactors(dd, randomData);

            Scaling scalingOp(dd, scaleFactors);

            THEN("apply and applyAdjoint yield the correct results")
            {
                auto resultApply = scalingOp.apply(input);
                for (index_t i = 0; i < resultApply.getSize(); ++i)
                    REQUIRE(resultApply[i] == Approx(inputScalar * scaleFactors[i]));

                auto resultApplyAdjoint = scalingOp.applyAdjoint(input);
                for (index_t i = 0; i < resultApply.getSize(); ++i)
                    REQUIRE(resultApplyAdjoint[i] == Approx(inputScalar * scaleFactors[i]));
            }
        }
    }
}
