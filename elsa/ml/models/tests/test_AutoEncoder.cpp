/**
 * @file test_AutoEncoder.cpp
 *
 * @brief Tests for the AutoEncoder class
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

TEST_SUITE_BEGIN("ml_models");

TEST_CASE_TEMPLATE("AutoEncoder: Testing construction", data_t, float, double)
{
    GIVEN("a DataDescriptor")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 45, 11, 7;
        VolumeDescriptor volDescr(numCoeff);

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


TEST_SUITE_END();
