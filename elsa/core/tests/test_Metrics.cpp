/**
 * @file test_Metrics.cpp
 *
 * @brief Tests for Metrics header
 *
 * @author Andi Braimllari
 */

#include "Metrics.hpp"
#include "DataContainer.h"
#include "VolumeDescriptor.h"

#include "doctest/doctest.h"
#include <testHelpers.h>

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("core");

TEST_CASE_TEMPLATE("Metrics: Testing the metrics", TestType, float, double)
{
    GIVEN("a DataContainer")
    {
        IndexVector_t sizeVector(2);
        sizeVector << 2, 4;
        VolumeDescriptor volDescr(sizeVector);

        DataContainer<TestType> dataCont1(volDescr);
        dataCont1 = 5;
        DataContainer<TestType> dataCont2(volDescr);
        dataCont2 = 8;

        WHEN("running the Relative Error")
        {
            long double relErr = relativeError<TestType>(dataCont1, dataCont2);
            THEN("it produces the correct result")
            {
                REQUIRE_UNARY(checkApproxEq(relErr, 3.0 / 8));
            }
        }

        WHEN("running the Peak Signal-to-Noise Ratio")
        {
            long double psnr = peakSignalToNoiseRatio<TestType>(dataCont1, dataCont2, 255);
            THEN("it produces the correct result") { REQUIRE_UNARY(checkApproxEq(psnr, 38.58837)); }
        }
    }
}

TEST_SUITE_END();
