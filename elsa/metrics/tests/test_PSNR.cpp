/**
 * @file test_PSNR.cpp
 *
 * @brief Tests for the PSNR class
 *
 * @author Andi Braimllari
 */

#include "Error.h"
#include "PSNR.h"
#include "VolumeDescriptor.h"

#include "doctest/doctest.h"
#include <testHelpers.h>

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("metrics");

TEST_CASE_TEMPLATE("PSNR: Testing construction", data_t, float, double)
{
    GIVEN("a DataDescriptor")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 45, 11, 7;
        VolumeDescriptor volDescr(numCoeff);

        WHEN("instantiating a PSNR operator")
        {
            THEN("the DataDescriptors are equal") {}
        }

        WHEN("cloning a PSNR operator")
        {
            THEN("cloned PSNR operator equals original PSNR operator") {}
        }
    }
}

TEST_SUITE_END();
