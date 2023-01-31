#include <doctest/doctest.h>
#include <limits>

#include "DataContainer.h"
#include "StrongTypes.h"
#include "testHelpers.h"
#include "VolumeDescriptor.h"
#include "ProximalBoxConstraint.h"

using namespace elsa;
using namespace doctest;

TYPE_TO_STRING(complex<float>);
TYPE_TO_STRING(complex<double>);

TEST_SUITE_BEGIN("functionals");

TEST_CASE_TEMPLATE("IndicatorBox: Default constructing box constraint", data_t, float, double)
{
    ProximalBoxConstraint<data_t> prox;

    GIVEN("Some 1D DataContainer")
    {
        VolumeDescriptor desc({15});
        DataContainer<data_t> x(desc);
        x = 4;

        THEN("it is unchanged from the proximal operator")
        {
            auto v = prox.apply(x, geometry::Threshold<data_t>(1));

            CHECK_UNARY(isApprox(x, v));
        }
    }

    GIVEN("Some 2D DataContainer")
    {
        VolumeDescriptor desc({15, 4});
        DataContainer<data_t> x(desc);
        x = 123;

        THEN("it is unchanged from the proximal operator")
        {
            auto v = prox.apply(x, geometry::Threshold<data_t>(1));

            CHECK_UNARY(isApprox(x, v));
        }
    }
}

TEST_CASE_TEMPLATE("IndicatorBox: Constructing non-negativity constraint", data_t, float, double)
{
    ProximalBoxConstraint<data_t> prox(data_t{0.f});

    GIVEN("Some 1D DataContainer")
    {
        VolumeDescriptor desc({15});
        DataContainer<data_t> x(desc);

        for (int i = 0; i < 15; ++i) {
            x[i] = data_t{i - 5.f};
        }

        THEN("it is unchanged from the proximal operator")
        {
            auto v = prox.apply(x, geometry::Threshold<data_t>(1));

            for (int i = 0; i < 5; ++i) {
                CHECK_EQ(v[i], doctest::Approx(0));
            }

            for (int i = 5; i < 15; ++i) {
                CHECK_EQ(v[i], doctest::Approx(x[i]));
            }
        }
    }
}

TEST_CASE_TEMPLATE("IndicatorBox: Constructing box constraint", data_t, float, double)
{
    ProximalBoxConstraint<data_t> prox(data_t{-3.f}, data_t{3.f});

    GIVEN("Some 1D DataContainer")
    {
        VolumeDescriptor desc({15});
        DataContainer<data_t> x(desc);

        for (int i = 0; i < 15; ++i) {
            x[i] = data_t{i - 8.f};
        }

        THEN("it is unchanged from the proximal operator")
        {
            auto v = prox.apply(x, geometry::Threshold<data_t>(1));
            CAPTURE(x);
            CAPTURE(v);

            for (int i = 0; i < 5; ++i) {
                CHECK_EQ(v[i], doctest::Approx(-3));
            }

            for (int i = 5; i < 11; ++i) {
                CHECK_EQ(v[i], doctest::Approx(x[i]));
            }

            for (int i = 11; i < 15; ++i) {
                CHECK_EQ(v[i], doctest::Approx(3));
            }
        }
    }
}
