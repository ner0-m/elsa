/**
 * @file test_Scaling.cpp
 *
 * @brief Tests for Scaling operator class
 *
 * @author Matthias Wieczorek - initial code
 * @author David Frank - rewrite
 * @author Tobias Lasser - minor extensions
 */

#include "doctest/doctest.h"
#include "Scaling.h"
#include "VolumeDescriptor.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("core");

TEST_CASE_TEMPLATE("Scaling: Testing construction", data_t, float, double)
{
    GIVEN("a descriptor")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 11, 17;
        VolumeDescriptor dd(numCoeff);

        WHEN("instantiating an isotropic scaling operator")
        {
            real_t scaleFactor = 3.5f;
            Scaling<data_t> scalingOp(dd, scaleFactor);

            THEN("the descriptors are as expected")
            {
                REQUIRE(scalingOp.getDomainDescriptor() == dd);
                REQUIRE(scalingOp.getRangeDescriptor() == dd);
            }

            THEN("the scaling is isotropic and correct")
            {
                REQUIRE(scalingOp.isIsotropic());
                REQUIRE(scalingOp.getScaleFactor() == scaleFactor);
                REQUIRE_THROWS_AS(scalingOp.getScaleFactors(), LogicError);
            }
        }

        WHEN("instantiating an anisotropic scaling operator")
        {
            DataContainer<data_t> dc(dd);
            dc = 3.5f;
            Scaling<data_t> scalingOp(dd, dc);

            THEN("the descriptors  are as expected")
            {
                REQUIRE(scalingOp.getDomainDescriptor() == dd);
                REQUIRE(scalingOp.getRangeDescriptor() == dd);
            }

            THEN("the scaling is anisotropic")
            {
                REQUIRE(!scalingOp.isIsotropic());
                REQUIRE(scalingOp.getScaleFactors() == dc);
                REQUIRE_THROWS_AS(scalingOp.getScaleFactor(), LogicError);
            }
        }

        WHEN("cloning a scaling operator")
        {
            Scaling<data_t> scalingOp(dd, 3.5f);
            auto scalingOpClone = scalingOp.clone();

            THEN("everything matches")
            {
                REQUIRE(scalingOpClone.get() != &scalingOp);
                REQUIRE(*scalingOpClone == scalingOp);
            }
        }
    }
}

TEST_CASE_TEMPLATE("Scaling: Testing apply to data", data_t, float, double)
{
    GIVEN("some data")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 34, 13;
        VolumeDescriptor dd(numCoeff);
        DataContainer<data_t> input(dd);
        data_t inputScalar{2.5};
        input = inputScalar;

        WHEN("applying isotropic scaling")
        {
            real_t scaleFactor{3.7f};
            Scaling<data_t> scalingOp(dd, scaleFactor);

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
            Vector_t<data_t> randomData(dd.getNumberOfCoefficients());
            randomData.setRandom();
            DataContainer<data_t> scaleFactors(dd, randomData);

            Scaling<data_t> scalingOp(dd, scaleFactors);

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
TEST_SUITE_END();
