#include <doctest/doctest.h>
#include <limits>

#include "DataContainer.h"
#include "testHelpers.h"
#include "VolumeDescriptor.h"
#include "IndicatorFunctionals.h"

using namespace elsa;
using namespace doctest;

TYPE_TO_STRING(complex<float>);
TYPE_TO_STRING(complex<double>);

TEST_SUITE_BEGIN("functionals");

TEST_CASE_TEMPLATE("IndicatorBox: Testing", data_t, float, double)
{
    VolumeDescriptor desc({7});

    GIVEN("An indicator box with lower = upper = infinity")
    {
        IndicatorBox<data_t> box(desc);

        DataContainer<data_t> x(desc);
        x = 345;

        CHECK_EQ(box.evaluate(x), data_t{0});

        WHEN("Creating a copy")
        {
            auto clone = box.clone();

            CHECK_EQ(*clone, box);
            CHECK_EQ(clone->evaluate(x), data_t{0});
        }
    }

    GIVEN("An indicator box with specific lower and uppwer bound")
    {
        IndicatorBox<data_t> box(desc, 3, 6);

        DataContainer<data_t> x(desc);
        x = 4;

        CHECK_EQ(box.evaluate(x), data_t{0});

        // Some other values inside the region
        x[0] = 3;
        x[4] = 5;
        x[5] = 6;

        CHECK_EQ(box.evaluate(x), data_t{0});

        // Some value outside the range
        x[1] = data_t{6.1f};
        CHECK_EQ(box.evaluate(x), std::numeric_limits<data_t>::infinity());

        // Some value outside the range
        x[1] = 2;
        CHECK_EQ(box.evaluate(x), std::numeric_limits<data_t>::infinity());

        WHEN("Creating a copy")
        {
            auto clone = box.clone();

            CHECK_EQ(*clone, box);
            CHECK_EQ(clone->evaluate(x), std::numeric_limits<data_t>::infinity());
        }
    }
}

TEST_CASE_TEMPLATE("IndicatorNonNegativity: Testing", data_t, float, double)
{
    VolumeDescriptor desc({7});

    GIVEN("An indicator box with lower = upper = infinity")
    {
        IndicatorNonNegativity<data_t> nonneg(desc);

        DataContainer<data_t> x(desc);
        x = 345;

        CHECK_EQ(nonneg.evaluate(x), 0);

        WHEN("Creating a copy")
        {
            auto clone = nonneg.clone();

            CHECK_EQ(*clone, nonneg);
            CHECK_EQ(clone->evaluate(x), 0);
        }

        x[0] = -1;
        CHECK_EQ(nonneg.evaluate(x), std::numeric_limits<data_t>::infinity());

        WHEN("Creating a copy")
        {
            auto clone = nonneg.clone();

            CHECK_EQ(*clone, nonneg);
            CHECK_EQ(clone->evaluate(x), std::numeric_limits<data_t>::infinity());
        }
    }
}
