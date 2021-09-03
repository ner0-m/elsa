/**
 * @file test_MSE.cpp
 *
 * @brief Tests for the MSE class
 *
 * @author Andi Braimllari
 */

#include "Error.h"
#include "MSE.h"
#include "VolumeDescriptor.h"

#include "doctest/doctest.h"
#include <testHelpers.h>

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("metrics");

TEST_CASE_TEMPLATE("MSE: Testing construction", data_t, float, double)
{
    GIVEN("a DataDescriptor")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 8, 4, 52;
        VolumeDescriptor volDescr(numCoeff);

        WHEN("instantiating an MSE operator")
        {
            THEN("the DataDescriptors are equal") {}
        }

        WHEN("cloning an MSE operator")
        {
            THEN("cloned MSE operator equals original MSE operator") {}
        }
    }
}

TEST_SUITE_END();
