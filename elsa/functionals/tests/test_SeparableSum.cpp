#include <doctest/doctest.h>

#include "DataContainer.h"
#include "DataDescriptor.h"
#include "Functional.h"
#include "IdenticalBlocksDescriptor.h"
#include "L1Norm.h"
#include "VolumeDescriptor.h"
#include "testHelpers.h"
#include "SeparableSum.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("functionals");

TEST_CASE_TEMPLATE("SeparableSum: Testing with a single functional", data_t, float, double)
{
    VolumeDescriptor desc({5});

    L1Norm<data_t> l1(desc);
    SeparableSum<data_t> sepsum(l1);

    IdenticalBlocksDescriptor blockdesc(1, desc);

    CHECK_EQ(sepsum.getDomainDescriptor(), blockdesc);

    DataContainer<data_t> x(blockdesc);
    x = 1;

    CHECK_EQ(sepsum.evaluate(x), l1.evaluate(x));

    THEN("Clone is equal to original")
    {
        auto clone = sepsum.clone();

        CHECK_EQ(*clone, sepsum);
        CHECK_EQ(clone->getDomainDescriptor(), blockdesc);
    }
}

TEST_CASE_TEMPLATE("SeparableSum: Testing with a two functional", data_t, float, double)
{
    VolumeDescriptor desc({5});

    L1Norm<data_t> l11(desc);
    L1Norm<data_t> l12(desc);

    SeparableSum<data_t> sepsum(l11, l12);
    IdenticalBlocksDescriptor blockdesc(2, desc);

    CHECK_EQ(sepsum.getDomainDescriptor(), blockdesc);

    DataContainer<data_t> x(blockdesc);
    x = 1;

    DataContainer<data_t> y(desc);
    y = 1;

    CHECK_EQ(sepsum.evaluate(x), l11.evaluate(y) + l12.evaluate(y));

    THEN("Clone is equal to original")
    {
        auto clone = sepsum.clone();

        CHECK_EQ(*clone, sepsum);
        CHECK_EQ(clone->getDomainDescriptor(), blockdesc);
    }
}

TEST_CASE_TEMPLATE("SeparableSum: Testing with a three functional", data_t, float, double)
{
    VolumeDescriptor desc({5});

    L1Norm<data_t> l11(desc);
    L1Norm<data_t> l12(desc);
    L1Norm<data_t> l13(desc);

    SeparableSum<data_t> sepsum(l11, l12, l13);
    IdenticalBlocksDescriptor blockdesc(3, desc);

    CHECK_EQ(sepsum.getDomainDescriptor(), blockdesc);

    DataContainer<data_t> x(blockdesc);
    x = 1;

    DataContainer<data_t> y(desc);
    y = 1;

    CHECK_EQ(sepsum.evaluate(x), l11.evaluate(y) + l12.evaluate(y) + l13.evaluate(y));

    THEN("Clone is equal to original")
    {
        auto clone = sepsum.clone();

        CHECK_EQ(*clone, sepsum);
        CHECK_EQ(clone->getDomainDescriptor(), blockdesc);
    }
}

TEST_CASE_TEMPLATE("SeparableSum: Testing with a four functional", data_t, float, double)
{
    VolumeDescriptor desc({5});

    L1Norm<data_t> l11(desc);
    L1Norm<data_t> l12(desc);
    L1Norm<data_t> l13(desc);
    L1Norm<data_t> l14(desc);

    SeparableSum<data_t> sepsum(l11, l12, l13, l14);
    IdenticalBlocksDescriptor blockdesc(4, desc);

    CHECK_EQ(sepsum.getDomainDescriptor(), blockdesc);

    DataContainer<data_t> x(blockdesc);
    x = 1;

    DataContainer<data_t> y(desc);
    y = 1;

    CHECK_EQ(sepsum.evaluate(x),
             l11.evaluate(y) + l12.evaluate(y) + l13.evaluate(y) + l14.evaluate(y));

    THEN("Clone is equal to original")
    {
        auto clone = sepsum.clone();

        CHECK_EQ(*clone, sepsum);
        CHECK_EQ(clone->getDomainDescriptor(), blockdesc);
    }
}
