/**
 * @file test_Convolution.cpp
 *
 * @brief Tests for the convolution operator.
 *
 * @author Jonas Jelten - initial code
 */

#include "doctest/doctest.h"
#include "Convolution.h"
#include "VolumeDescriptor.h"
#include "testHelpers.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("core");

TEST_CASE_TEMPLATE("Convolution: Testing construction", data_t, float, double)
{
    GIVEN("a descriptor")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 11, 17;
        VolumeDescriptor dd(numCoeff);

        WHEN("instantiating an isotropic scaling operator")
        {
            Convolution<data_t> scalingOp(dd);
        }
    }
}

TEST_SUITE_END();
