/**
 * @file test_AutoEncoder.cpp
 *
 * @brief Tests for the AutoEncoder class
 *
 * @author Andi Braimllari
 */

#include "AutoEncoder.h"
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
        // ...

        WHEN("instantiating an AutoEncoder model")
        {
            ml::AutoEncoder<real_t, elsa::ml::MlBackend::Cudnn> aeModel(VolumeDescriptor{{28, 28}},
                                                                        32);

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

TEST_CASE_TEMPLATE("AutoEncoder: Pre-training testing", data_t, float, double)
{
    GIVEN("a DataDescriptor")
    {
        // ...

        WHEN("instantiating an AutoEncoder model")
        {
            // ...

            THEN("the spatial dimensions of the input match to those of the output")
            {
                // ...
            }
        }
    }
}

TEST_SUITE_END();
