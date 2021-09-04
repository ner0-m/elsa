/**
 * @file test_RelativeError.cpp
 *
 * @brief Tests for the RelativeError class
 *
 * @author Andi Braimllari
 */

#include "Error.h"
#include "RelativeError.h"
#include "VolumeDescriptor.h"

#include "doctest/doctest.h"
#include <testHelpers.h>

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("metrics");

TEST_CASE_TEMPLATE("RelativeError: Testing construction", data_t, float, double)
{
    GIVEN("a DataDescriptor")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 8, 4, 52;
        VolumeDescriptor volDescr(numCoeff);

        WHEN("instantiating an RelativeError operator")
        {
            THEN("the DataDescriptors are equal") {}
        }

        WHEN("cloning an RelativeError operator")
        {
            THEN("cloned RelativeError operator equals original RelativeError operator") {}
        }
    }
}

TEST_SUITE_END();
