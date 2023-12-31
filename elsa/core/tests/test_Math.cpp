#include "doctest/doctest.h"

#include "Math.hpp"

TEST_SUITE_BEGIN("Math");

using namespace elsa;

TEST_CASE("Math::factorial")
{
    CHECK_EQ(1, math::factorial(0));
    CHECK_EQ(1, math::factorial(1));

    auto fac = 1;
    for (int i = 1; i < 10; ++i) {
        fac *= i;
        CHECK_EQ(fac, math::factorial(i));
    }
}

TEST_CASE("Math::binom")
{
    GIVEN("n == 10")
    {
        const index_t n = 10;

        WHEN("k larger than n")
        {
            CHECK_EQ(math::binom(n, 15), 0);
        }

        WHEN("k == 0 or k == n")
        {
            CHECK_EQ(math::binom(n, 0), 1);
            CHECK_EQ(math::binom(n, 10), 1);
        }

        CHECK_EQ(math::binom(n, 1), 10);
        CHECK_EQ(math::binom(n, 2), 45);
        CHECK_EQ(math::binom(n, 3), 120);
        CHECK_EQ(math::binom(n, 4), 210);
        CHECK_EQ(math::binom(n, 5), 252);
        CHECK_EQ(math::binom(n, 6), 210);
        CHECK_EQ(math::binom(n, 7), 120);
        CHECK_EQ(math::binom(n, 8), 45);
        CHECK_EQ(math::binom(n, 9), 10);
    }
}

TEST_CASE("Math::heaviside")
{
    constexpr index_t size = 200;
    constexpr real_t c = 0.5;
    const auto linspace = Vector_t<real_t>::LinSpaced(size, -2, 2);

    for (std::size_t i = 0; i < size; ++i) {
        auto res = math::heaviside(linspace[i], c);
        if (linspace[i] == 0.) {
            CHECK_EQ(res, c);
        } else if (linspace[i] < 0) {
            CHECK_EQ(res, 0);
        } else if (linspace[i] > 0) {
            CHECK_EQ(res, 1);
        }
    }
}

TEST_SUITE_END();
