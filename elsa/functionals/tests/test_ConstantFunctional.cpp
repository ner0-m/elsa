#include <doctest/doctest.h>

#include "DataContainer.h"
#include "testHelpers.h"
#include "ConstantFunctional.h"
#include "VolumeDescriptor.h"

using namespace elsa;
using namespace doctest;

TYPE_TO_STRING(complex<float>);
TYPE_TO_STRING(complex<double>);

TEST_SUITE_BEGIN("functionals");

TEST_CASE_TEMPLATE("ConstantFunctional: Clone and Compare", data_t, float, complex<float>, double,
                   complex<double>)
{
    GIVEN("two 1D Functionals with the same underlying domain")
    {
        IndexVector_t numCoeff(1);
        numCoeff << 4;
        VolumeDescriptor dd(numCoeff);

        ConstantFunctional<data_t> cfunc1(dd, 5);
        ConstantFunctional<data_t> cfunc2(dd, 6);

        CHECK_NE(cfunc1, cfunc2);

        WHEN("Cloing one, they should compare equal")
        {
            auto clone1 = cfunc1.clone();
            CHECK_EQ(cfunc1, *clone1);

            auto clone2 = cfunc2.clone();
            CHECK_EQ(cfunc2, *clone2);

            CHECK_NE(*clone1, *clone2);
        }
    }

    GIVEN("two 1D Functionals with the different domains")
    {
        IndexVector_t numCoeff1({{4}});
        VolumeDescriptor dd1(numCoeff1);

        IndexVector_t numCoeff2({{6}});
        VolumeDescriptor dd2(numCoeff2);

        ConstantFunctional<data_t> cfunc1(dd1, 5);
        ConstantFunctional<data_t> cfunc2(dd2, 5);

        CHECK_NE(cfunc1, cfunc2);

        WHEN("Cloing one, they should compare equal")
        {
            auto clone1 = cfunc1.clone();
            CHECK_EQ(cfunc1, *clone1);

            auto clone2 = cfunc2.clone();
            CHECK_EQ(cfunc2, *clone2);

            CHECK_NE(*clone1, *clone2);
        }
    }
}

TEST_CASE_TEMPLATE("ConstantFunctional: Testing evaluation", data_t, float, complex<float>, double,
                   complex<double>)
{
    GIVEN("a 1D Functional")
    {
        IndexVector_t numCoeff(1);
        numCoeff << 4;
        VolumeDescriptor dd(numCoeff);

        ConstantFunctional<data_t> cfunc(dd, 5);

        DataContainer<data_t> x(dd);
        x = 1234;

        CHECK_EQ(cfunc.evaluate(x), data_t{5});
    }

    GIVEN("a 2D Functional")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 4, 5;
        VolumeDescriptor dd(numCoeff);

        ConstantFunctional<data_t> cfunc(dd, 6.7);

        DataContainer<data_t> x(dd);
        x = 1234;

        CHECK_EQ(cfunc.evaluate(x), data_t{6.7});
    }

    GIVEN("a 3D Functional")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 4, 5, 9;
        VolumeDescriptor dd(numCoeff);

        ConstantFunctional<data_t> cfunc(dd, 2);

        DataContainer<data_t> x(dd);
        x = 1234;

        CHECK_EQ(cfunc.evaluate(x), data_t{2});
    }
}

TEST_CASE_TEMPLATE("ZeroFunctional: Clone and Compare", data_t, float, complex<float>, double,
                   complex<double>)
{
    GIVEN("two 1D Functionals with the same underlying domain")
    {
        IndexVector_t numCoeff({{4}});
        VolumeDescriptor dd(numCoeff);

        ZeroFunctional<data_t> cfunc1(dd);
        ZeroFunctional<data_t> cfunc2(dd);

        CHECK_EQ(cfunc1, cfunc2);

        WHEN("Cloing one, they should compare equal")
        {
            auto clone1 = cfunc1.clone();
            CHECK_EQ(cfunc1, *clone1);

            auto clone2 = cfunc2.clone();
            CHECK_EQ(cfunc2, *clone2);

            CHECK_EQ(*clone1, *clone2);
        }
    }

    GIVEN("two 1D Functionals with the different domains")
    {
        IndexVector_t numCoeff1({{4}});
        VolumeDescriptor dd1(numCoeff1);

        IndexVector_t numCoeff2({{6}});
        VolumeDescriptor dd2(numCoeff2);

        ZeroFunctional<data_t> cfunc1(dd1);
        ZeroFunctional<data_t> cfunc2(dd2);

        CHECK_NE(cfunc1, cfunc2);

        WHEN("Cloing one, they should compare equal")
        {
            auto clone1 = cfunc1.clone();
            CHECK_EQ(cfunc1, *clone1);

            auto clone2 = cfunc2.clone();
            CHECK_EQ(cfunc2, *clone2);

            CHECK_NE(*clone1, *clone2);
        }
    }
}

TEST_CASE_TEMPLATE("ZeroFunctional", data_t, float, complex<float>, double, complex<double>)
{
    GIVEN("A 1D Functional")
    {
        IndexVector_t numCoeff(1);
        numCoeff << 4;
        VolumeDescriptor dd(numCoeff);

        ZeroFunctional<data_t> zeroFn(dd);

        DataContainer<data_t> x(dd);
        x = 1234;

        CHECK_EQ(zeroFn.evaluate(x), data_t{0});
    }

    GIVEN("A 2D Functional")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 4, 9;
        VolumeDescriptor dd(numCoeff);

        ZeroFunctional<data_t> zeroFn(dd);

        DataContainer<data_t> x(dd);
        x = 1234;

        CHECK_EQ(zeroFn.evaluate(x), data_t{0});
    }

    GIVEN("A 3D Functional")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 4, 9, 5;
        VolumeDescriptor dd(numCoeff);

        ZeroFunctional<data_t> zeroFn(dd);

        DataContainer<data_t> x(dd);
        x = 1234;

        CHECK_EQ(zeroFn.evaluate(x), data_t{0});
    }
}
