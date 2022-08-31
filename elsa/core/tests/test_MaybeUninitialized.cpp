#include "doctest/doctest.h"

#include "MaybeUninitialized.hpp"

TEST_CASE_TEMPLATE("MaybeUninitialized: Construction without value", data_t, float, double)
{
    elsa::MaybeUninitialized<data_t> uninit;

    CHECK_UNARY_FALSE(uninit.isInitialized());
}

TEST_CASE_TEMPLATE("MaybeUninitialized: Assign to uninitialized", data_t, float, double)
{
    elsa::MaybeUninitialized<data_t> uninit;

    CHECK_UNARY_FALSE(uninit.isInitialized());

    uninit = 5.0;

    CHECK_UNARY(uninit.isInitialized());
    CHECK_EQ(*uninit, data_t(5.0));
}

TEST_CASE_TEMPLATE("MaybeUninitialized: Construction with value", data_t, float, double)
{
    elsa::MaybeUninitialized<data_t> uninit(data_t(5.0));

    CHECK_UNARY(uninit.isInitialized());
    CHECK_EQ(*uninit, data_t(5.0));
}

TEST_CASE_TEMPLATE("MaybeUninitialized: Construction with value", data_t, float, double)
{
    data_t value = 12.0;
    elsa::MaybeUninitialized<data_t> uninit(value);

    CHECK_UNARY(uninit.isInitialized());
    CHECK_EQ(*uninit, value);
}

TEST_CASE_TEMPLATE("MaybeUninitialized: Compare equality", data_t, float, double)
{
    elsa::MaybeUninitialized<data_t> uninit1;
    elsa::MaybeUninitialized<data_t> uninit2;

    CHECK_EQ(uninit1, uninit1);
    CHECK_EQ(uninit1, uninit2);

    uninit1 = 4;

    CHECK_NE(uninit1, uninit2);
    CHECK_NE(uninit2, uninit1);

    uninit1 = 2;

    CHECK_NE(uninit1, uninit2);
    CHECK_NE(uninit2, uninit1);

    uninit1 = 6;
    uninit2 = 6;

    CHECK_EQ(uninit1, uninit2);
    CHECK_EQ(uninit2, uninit1);
}
