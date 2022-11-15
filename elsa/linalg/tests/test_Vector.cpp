#include "doctest/doctest.h"

#include "thrust/fill.h"
#include "Vector.h"
#include <algorithm>
#include <cstddef>

TEST_SUITE_BEGIN("linalg");

using namespace doctest;
using namespace elsa;

TEST_CASE_TEMPLATE("Construct vector from size", T, float, double)
{
    elsa::linalg::Vector<T> v(5);

    CHECK_EQ(5, v.size());

    WHEN("Assigning a value to the vector")
    {
        v = 1234;
        CHECK_UNARY(std::all_of(v.begin(), v.end(), [](auto x) { return x == 1234; }));
    }

    WHEN("Assigning via operator[]")
    {
        for (std::size_t i = 0; i < v.size(); ++i) {
            v[i] = i;
        }

        for (std::size_t i = 0; i < v.size(); ++i) {
            CHECK_EQ(i, v(i));
        }
    }
}

TEST_CASE_TEMPLATE("Construct vector from size and value", T, float, double)
{
    elsa::linalg::Vector<T> v(5, 12345);

    CHECK_EQ(5, v.size());

    CHECK_UNARY(std::all_of(v.begin(), v.end(), [](auto x) { return x == 12345; }));

    WHEN("Assigning a value to the vector")
    {
        v = 1234;
        CHECK_UNARY(std::all_of(v.begin(), v.end(), [](auto x) { return x == 1234; }));
    }

    WHEN("Assigning via operator[]")
    {
        for (std::size_t i = 0; i < v.size(); ++i) {
            v[i] = i;
        }

        for (std::size_t i = 0; i < v.size(); ++i) {
            CHECK_EQ(i, v(i));
        }
    }

    WHEN("adding scalar value")
    {
        v += 5;
        CHECK_UNARY(std::all_of(v.begin(), v.end(), [](auto x) { return x == 12350; }));

        auto w = v + 10;
        CHECK_UNARY(std::all_of(w.begin(), w.end(), [](auto x) { return x == 12360; }));

        auto u = 20 + v;
        CHECK_UNARY(std::all_of(u.begin(), u.end(), [](auto x) { return x == 12370; }));
    }

    WHEN("subtracting scalar value")
    {
        v -= 12345;
        CHECK_UNARY(std::all_of(v.begin(), v.end(), [](auto x) { return x == 0; }));

        auto w = v - 10;
        CHECK_UNARY(std::all_of(w.begin(), w.end(), [](auto x) { return x == -10; }));

        auto u = 20 - v;
        CHECK_UNARY(std::all_of(u.begin(), u.end(), [](auto x) { return x == -20; }));
    }

    WHEN("multiplying by scalar value")
    {
        v *= 2;
        CHECK_UNARY(std::all_of(v.begin(), v.end(), [](auto x) { return x == (12345 * 2); }));

        auto w = v * 2;
        CHECK_UNARY(std::all_of(w.begin(), w.end(), [](auto x) { return x == (12345 * 4); }));

        auto u = 0.25 * v;
        CHECK_UNARY(std::all_of(u.begin(), u.end(), [](auto x) { return x == (12345 * 0.5); }));
    }

    WHEN("dividing by scalar value")
    {
        v /= 4;
        CHECK_UNARY(std::all_of(v.begin(), v.end(), [](auto x) { return x == (12345 / 4.f); }));

        auto w = v / 2.;
        CHECK_UNARY(std::all_of(w.begin(), w.end(), [](auto x) { return x == (12345 / 8.f); }));
    }
}

TEST_CASE_TEMPLATE("Construct vector from size and value", T, float, double)
{
    elsa::linalg::Vector<T> v1(10, 2);
    elsa::linalg::Vector<T> v2(10, 1);

    CHECK_EQ(10, v1.size());
    CHECK_EQ(10, v2.size());

    WHEN("Adding two vectors")
    {
        auto v = v1 + v2;
        CHECK_UNARY(std::all_of(v.begin(), v.end(), [](auto x) { return x == 3; }));

        v1 += v2;
        CHECK_UNARY(std::all_of(v.begin(), v.end(), [](auto x) { return x == 3; }));
    }

    WHEN("Subtracting two vectors")
    {
        auto v = v1 - v2;
        CHECK_UNARY(std::all_of(v.begin(), v.end(), [](auto x) { return x == 1; }));

        v1 -= v2;
        CHECK_UNARY(std::all_of(v.begin(), v.end(), [](auto x) { return x == 1; }));
    }

    WHEN("Multiplying two vectors")
    {
        auto v = v1 * v2;
        CHECK_UNARY(std::all_of(v.begin(), v.end(), [](auto x) { return x == 2; }));

        v1 *= v2;
        CHECK_UNARY(std::all_of(v.begin(), v.end(), [](auto x) { return x == 2; }));
    }

    WHEN("Divide two vectors")
    {
        auto v = v2 / v1;
        CHECK_UNARY(std::all_of(v.begin(), v.end(), [](auto x) { return x == 0.5; }));

        v2 /= v1;
        CHECK_UNARY(std::all_of(v.begin(), v.end(), [](auto x) { return x == 0.5; }));
    }
}

TEST_SUITE_END();
