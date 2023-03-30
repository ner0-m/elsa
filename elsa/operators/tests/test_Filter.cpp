#include "Complex.h"
#include "doctest/doctest.h"
#include "Filter.h"
#include "VolumeDescriptor.h"
#include <cmath>

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("core");

TEST_CASE_TEMPLATE("makeFilter: Testing ramLak", data_t, float, double)
{
    GIVEN("a descriptor")
    {
        IndexVector_t numCoeff{2};
        numCoeff << 5, 128;
        VolumeDescriptor dd{numCoeff};

        WHEN("making a RamLak filter")
        {
            auto RL = makeRamLak(dd);
            THEN("the descriptors are as expected")
            {
                auto domainCoeffs = RL.getDomainDescriptor().getNumberOfCoefficientsPerDimension();
                auto rangeCoeffs = RL.getRangeDescriptor().getNumberOfCoefficientsPerDimension();
                REQUIRE_EQ(rangeCoeffs[0], 5);
                REQUIRE_EQ(rangeCoeffs[1], 1);
            }
            THEN("the coefficients are correct")
            {
                auto deltaK = 2 * pi_t / 5;
                auto filter = RL.getScaleFactors();

                REQUIRE_EQ(filter[0], 0.25 * deltaK);
                REQUIRE_EQ(filter[1], 2 * pi_t);
                REQUIRE_EQ(filter[2], 4 * pi_t);
                REQUIRE_EQ(filter[3], 4 * pi_t);
                REQUIRE_EQ(filter[4], 2 * pi_t);
            }
        }
    }
}

TEST_SUITE_END();
