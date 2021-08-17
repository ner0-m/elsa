/**
 * @file test_AutoEncoder.cpp
 *
 * @brief Tests for the AutoEncoder class
 *
 * @author Andi Braimllari
 */

#include "VolumeDescriptor.h"
#include "Error.h"

#include "doctest/doctest.h"
#include <testHelpers.h>

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("ml");

TEST_CASE_TEMPLATE("AutoEncoder: Testing construction", data_t, float, double)
{
    GIVEN("a DataDescriptor")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 45, 11, 7;
        VolumeDescriptor volDescr(numCoeff);

        WHEN("instantiating an AutoEncoder model")
        {
            // TODO create object here

            THEN("the DataDescriptors are equal")
            {
                //                REQUIRE_EQ(sThrOp.getRangeDescriptor(), volDescr);
            }
        }

        WHEN("cloning an AutoEncoder model")
        {
            // TODO clone object here

            THEN("cloned AutoEncoder model equals original AutoEncoder model")
            {
                //                REQUIRE_NE(sThrOpClone.get(), &sThrOp);
                //                REQUIRE_EQ(*sThrOpClone, sThrOp);
            }
        }
    }
}

TEST_SUITE_END();
