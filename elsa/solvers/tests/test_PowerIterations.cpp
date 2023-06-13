#include "Identity.h"
#include "Scaling.h"
#include "doctest/doctest.h"

#include "PowerIterations.h"
#include "VolumeDescriptor.h"
#include "testHelpers.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("problems");

TEST_CASE_TEMPLATE("PowerIterations: Approximating the eigenvalue of the idetity matrix", data_t,
                   float, double)
{
    GIVEN("The Identity operator")
    {
        VolumeDescriptor desc({5, 5});
        Identity<data_t> op(desc);

        THEN("The largest eigenvalue is 1")
        {
            auto eig = powerIterations(op);

            CHECK_EQ(eig, doctest::Approx(1));
        }
    }

    GIVEN("An operator scaling by 5.0")
    {
        VolumeDescriptor desc({5, 5});
        Scaling<data_t> op(desc, 5);

        THEN("The largest eigenvalue is 5")
        {
            auto eig = powerIterations(op);

            CHECK_EQ(eig, doctest::Approx(5));
        }
    }
}

TEST_SUITE_END();
