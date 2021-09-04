/**
 * @file test_PeakSignaltoNoiseRatio.cpp
 *
 * @brief Tests for the PeakSignaltoNoiseRatio class
 *
 * @author Andi Braimllari
 */

#include "Error.h"
#include "PeakSignaltoNoiseRatio.h"
#include "VolumeDescriptor.h"

#include "doctest/doctest.h"
#include <testHelpers.h>

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("metrics");

TEST_CASE_TEMPLATE("PeakSignaltoNoiseRatio: Testing construction", data_t, float, double)
{
    GIVEN("a DataDescriptor")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 45, 11, 7;
        VolumeDescriptor volDescr(numCoeff);

        WHEN("instantiating a PeakSignaltoNoiseRatio operator")
        {
            THEN("the DataDescriptors are equal") {}
        }

        WHEN("cloning a PeakSignaltoNoiseRatio operator")
        {
            THEN("cloned PeakSignaltoNoiseRatio operator equals original PeakSignaltoNoiseRatio "
                 "operator")
            {
            }
        }
    }
}

TEST_SUITE_END();
