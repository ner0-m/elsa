#include "doctest/doctest.h"

#include "CartesianIndices.h"

#include <iostream>
#include <sstream>
#include <vector>
#include "spdlog/fmt/fmt.h"
#include "spdlog/fmt/bundled/ranges.h"

TEST_SUITE_BEGIN("CartesianIndices");

using namespace elsa;

namespace doctest
{
    template <>
    struct StringMaker<CartesianIndices> {
        static String convert(const CartesianIndices& value)
        {
            return fmt::format("{}", value).c_str();
        }
    };
    template <typename T>
    struct StringMaker<std::vector<T>> {
        static String convert(const std::vector<T>& value)
        {
            return fmt::format("{}", value).c_str();
        }
    };
    template <>
    struct StringMaker<IndexVector_t> {
        static String convert(const IndexVector_t& value)
        {
            Eigen::IOFormat format(4, 0, ", ", "", "", "", "[", "]");
            std::stringstream stream;
            stream << value.format(format);
            return stream.str().c_str();
        }
    };
} // namespace doctest

TEST_CASE("Begin and End iterator comparison")
{
    GIVEN("a 1D grid")
    {
        const CartesianIndices grid(std::vector<int>{5});

        CAPTURE(grid);

        auto begin = grid.begin();
        auto end = grid.end();

        CHECK_NE(begin, end);
        CHECK_EQ(end, end);
        CHECK_EQ(begin, begin);

        begin += 5;

        CAPTURE(*begin);
        CAPTURE(*end);
        CHECK_EQ(begin, end);
    }

    GIVEN("a 2D grid")
    {
        const CartesianIndices grid(std::vector<int>{5, 5});

        CAPTURE(grid);

        auto begin = grid.begin();
        auto end = grid.end();

        CHECK_NE(begin, end);
        CHECK_EQ(end, end);
        CHECK_EQ(begin, begin);

        begin += 25;

        CAPTURE(*begin);
        CAPTURE(*end);
        CHECK_EQ(begin, end);
    }
}

TEST_CASE("Construct 1D")
{
    const CartesianIndices grid(std::vector<int>{5});

    CAPTURE(grid);

    THEN("The sizes are correct")
    {
        CHECK_EQ(grid.dims(), 1);
        CHECK_EQ(grid.size(), 5);
    }

    THEN("The first and last coordinate are correct")
    {
        CHECK_EQ(grid.first(), IndexVector_t({{0}}));
        CHECK_EQ(grid.last(), IndexVector_t({{5}}));
    }

    WHEN("iterating over the grid using an indces")
    {
        auto begin = grid.begin();
        for (int i = 0; i < grid.size(); ++i) {
            CHECK_EQ(begin[i], IndexVector_t({{i}}));
        }
    }

    WHEN("iterating over the grid")
    {
        auto begin = grid.begin();
        auto end = grid.end();

        CHECK_NE(begin, end);

        auto pos = 0;
        for (; begin != end; ++begin) {
            CHECK_EQ(*begin, IndexVector_t({{pos++}}));
        }

        CHECK_EQ(pos, 5);
        CHECK_EQ(begin, end);
    }

    WHEN("increment and decrement iterator")
    {
        auto begin = grid.begin();

        CHECK_EQ(*begin, IndexVector_t({{0}}));

        // Prefix increment
        ++begin;
        CHECK_EQ(*begin, IndexVector_t({{1}}));

        // Postfix increment
        begin++;
        CHECK_EQ(*begin, IndexVector_t({{2}}));

        // Prefix decrement
        --begin;
        CHECK_EQ(*begin, IndexVector_t({{1}}));

        // Postfix decrement
        begin--;
        CHECK_EQ(*begin, IndexVector_t({{0}}));
        CHECK_EQ(*begin, *grid.begin());
    }

    WHEN("advance by n")
    {
        auto begin = grid.begin();

        begin += 0;
        CHECK_EQ(*begin, IndexVector_t({{0}}));

        begin += 2;
        CHECK_EQ(*begin, IndexVector_t({{2}}));

        begin -= 0;
        CHECK_EQ(*begin, IndexVector_t({{2}}));

        begin -= 2;
        CHECK_EQ(*begin, *grid.begin());
    }

    WHEN("Calculating distance")
    {
        auto begin = grid.begin();

        CHECK_EQ(begin - begin, 0);
        CHECK_EQ(std::distance(begin, begin), 0);

        auto iter = begin;
        ++iter;

        CHECK_EQ(begin - iter, -1);
        CHECK_EQ(std::distance(iter, begin), -1);

        CHECK_EQ(iter - begin, 1);
        CHECK_EQ(std::distance(begin, iter), 1);

        iter += grid.size() - 1;
        CHECK_EQ(iter - begin, grid.size());
    }
}

TEST_CASE("Construct 1D given start and end")
{
    CartesianIndices grid(std::vector<int>{5}, std::vector<int>{10});

    CAPTURE(grid);
    THEN("The sizes are correct")
    {
        CHECK_EQ(grid.dims(), 1);
        CHECK_EQ(grid.size(), 5);
    }

    THEN("The first and last coordinate are correct")
    {
        CHECK_EQ(grid.first(), IndexVector_t({{5}}));
        CHECK_EQ(grid.last(), IndexVector_t({{10}}));
    }

    WHEN("iterating over the grid")
    {
        auto begin = grid.begin();
        auto end = grid.end();

        CHECK_NE(begin, end);

        auto pos = 5;
        for (; begin != end; ++begin) {
            CHECK_EQ(*begin, IndexVector_t({{pos++}}));
        }

        CHECK_EQ(pos, 10);
        CHECK_EQ(begin, end);
    }

    WHEN("Calculating distance")
    {
        auto begin = grid.begin();

        CHECK_EQ(begin - begin, 0);
        CHECK_EQ(std::distance(begin, begin), 0);

        auto iter = begin;
        ++iter;

        CHECK_EQ(begin - iter, -1);
        CHECK_EQ(std::distance(iter, begin), -1);

        CHECK_EQ(iter - begin, 1);
        CHECK_EQ(std::distance(begin, iter), 1);

        iter += grid.size() - 1;
        CHECK_EQ(iter - begin, grid.size());
    }
}

TEST_CASE("Construct 2D")
{
    const CartesianIndices grid(std::vector<int>{5, 7});

    CAPTURE(grid);
    THEN("The sizes are correct")
    {
        CHECK_EQ(grid.dims(), 2);
        CHECK_EQ(grid.size(), 5 * 7);
    }

    THEN("The first and last coordinate are correct")
    {
        CHECK_EQ(grid.first(), IndexVector_t({{0, 0}}));
        CHECK_EQ(grid.last(), IndexVector_t({{5, 7}}));
    }

    WHEN("iterating over the grid")
    {
        auto begin = grid.begin();
        auto end = grid.end();

        CHECK_NE(begin, end);

        auto pos = 0;
        for (; begin != end; ++begin) {
            // I don't really want to repeat the logic completely again, but as
            // long as we in total traverse the same amount as size() of the
            // grid, it should be fine
            ++pos;
        }

        CHECK_EQ(pos, grid.size());
        CHECK_EQ(begin, end);
    }

    WHEN("increment iterator")
    {
        auto begin = grid.begin();
        CHECK_EQ(*begin, IndexVector_t({{0, 0}}));

        THEN("Last dimension is incremented")
        {
            ++begin;
            CHECK_EQ(*begin, IndexVector_t({{0, 1}}));

            ++begin;
            CHECK_EQ(*begin, IndexVector_t({{0, 2}}));
        }
    }

    THEN("Iterating over all indices")
    {
        auto begin = grid.begin();

        auto x1 = 0;
        auto x2 = 0;

        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 7; ++j) {
                CHECK_EQ(*begin, IndexVector_t({{x1, x2}}));
                ++begin;
                ++x2;
            }
            x2 = 0;
            ++x1;
        }
    }

    WHEN("advance by n")
    {
        auto begin = grid.begin();

        begin += 0;
        CHECK_EQ(*begin, IndexVector_t({{0, 0}}));

        begin += 2;
        CHECK_EQ(*begin, IndexVector_t({{0, 2}}));

        begin -= 0;
        CHECK_EQ(*begin, IndexVector_t({{0, 2}}));

        begin -= 2;
        CHECK_EQ(*begin, *grid.begin());

        begin += 7;
        CHECK_EQ(*begin, IndexVector_t({{1, 0}}));

        begin += 9;
        CHECK_EQ(*begin, IndexVector_t({{2, 2}}));

        begin -= 16;
        CHECK_EQ(*begin, IndexVector_t({{0, 0}}));
    }

    WHEN("Calculating distance")
    {
        auto begin = grid.begin();

        CHECK_EQ(begin - begin, 0);
        CHECK_EQ(std::distance(begin, begin), 0);

        auto iter = begin;
        ++iter;

        CHECK_EQ(begin - iter, -1);
        CHECK_EQ(std::distance(iter, begin), -1);

        CHECK_EQ(iter - begin, 1);
        CHECK_EQ(std::distance(begin, iter), 1);

        iter += grid.size() - 1;
        CHECK_EQ(iter - begin, grid.size());
    }
}

TEST_CASE("Construct 2D given start and end")
{
    CartesianIndices grid(std::vector<int>{5, 7}, std::vector<int>{10, 12});

    CAPTURE(grid);
    THEN("The sizes are correct")
    {
        CHECK_EQ(grid.dims(), 2);
        CHECK_EQ(grid.size(), 5 * 5);
    }

    THEN("The first and last coordinate are correct")
    {
        CHECK_EQ(grid.first(), IndexVector_t({{5, 7}}));
        CHECK_EQ(grid.last(), IndexVector_t({{10, 12}}));
    }

    WHEN("iterating over the grid")
    {
        auto begin = grid.begin();
        auto end = grid.end();

        CHECK_NE(begin, end);

        auto pos = 0;
        for (; begin != end; ++begin) {
            // I don't really want to repeat the logic completely again, but as
            // long as we in total traverse the same amount as size() of the
            // grid, it should be fine
            ++pos;
            CAPTURE(*begin);
            CAPTURE(*end);

            // Protect against endless loops
            REQUIRE(pos < grid.size() + 1);
        }

        CHECK_EQ(pos, grid.size());
        CHECK_EQ(begin, end);
    }
}

TEST_CASE("Construct 3D")
{
    const CartesianIndices grid(std::vector<int>{5, 7, 9});

    CAPTURE(grid);
    THEN("The sizes are correct")
    {
        CHECK_EQ(grid.dims(), 3);
        CHECK_EQ(grid.size(), 5 * 7 * 9);
    }

    THEN("The first and last coordinate are correct")
    {
        CHECK_EQ(grid.first(), IndexVector_t({{0, 0, 0}}));
        CHECK_EQ(grid.last(), IndexVector_t({{5, 7, 9}}));
    }

    WHEN("iterating over the grid")
    {
        auto begin = grid.begin();
        auto end = grid.end();

        CHECK_NE(begin, end);

        auto pos = 0;
        for (; begin != end; ++begin) {
            // I don't really want to repeat the logic completely again, but as
            // long as we in total traverse the same amount as size() of the
            // grid, it should be fine
            ++pos;

            CAPTURE(*begin);
            CAPTURE(*end);

            // Protect against endless loops
            REQUIRE(pos < grid.size() + 1);
        }

        CHECK_EQ(pos, grid.size());
        CHECK_EQ(begin, end);
    }

    THEN("Iterating over all indices")
    {
        auto begin = grid.begin();

        auto x1 = 0;
        auto x2 = 0;
        auto x3 = 0;

        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 7; ++j) {
                for (int k = 0; k < 9; ++k) {
                    CHECK_EQ(*begin, IndexVector_t({{x1, x2, x3}}));
                    ++begin;
                    ++x3;
                }
                x3 = 0;
                ++x2;
            }
            x2 = 0;
            ++x1;
        }
    }

    WHEN("advance by n")
    {
        auto begin = grid.begin();

        begin += 0;
        CHECK_EQ(*begin, IndexVector_t({{0, 0, 0}}));

        begin += 2;
        CHECK_EQ(*begin, IndexVector_t({{0, 0, 2}}));

        begin -= 0;
        CHECK_EQ(*begin, IndexVector_t({{0, 0, 2}}));

        begin -= 2;
        CHECK_EQ(*begin, *grid.begin());

        begin += 9;
        CHECK_EQ(*begin, IndexVector_t({{0, 1, 0}}));

        begin += 11;
        CHECK_EQ(*begin, IndexVector_t({{0, 2, 2}}));

        begin -= 20;
        CHECK_EQ(*begin, IndexVector_t({{0, 0, 0}}));
    }

    WHEN("Calculating distance")
    {
        auto begin = grid.begin();

        CHECK_EQ(begin - begin, 0);
        CHECK_EQ(std::distance(begin, begin), 0);

        auto iter = begin;
        ++iter;

        CHECK_EQ(begin - iter, -1);
        CHECK_EQ(std::distance(iter, begin), -1);

        CHECK_EQ(iter - begin, 1);
        CHECK_EQ(std::distance(begin, iter), 1);

        iter += grid.size() - 1;
        CHECK_EQ(iter - begin, grid.size());
    }
}

TEST_CASE("Construct 3D given start and end")
{
    CartesianIndices grid(std::vector<int>{5, 7, 9}, std::vector<int>{10, 12, 14});

    CAPTURE(grid);
    THEN("The sizes are correct")
    {
        CHECK_EQ(grid.dims(), 3);
        CHECK_EQ(grid.size(), 5 * 5 * 5);
    }

    THEN("The first and last coordinate are correct")
    {
        CHECK_EQ(grid.first(), IndexVector_t({{5, 7, 9}}));
        CHECK_EQ(grid.last(), IndexVector_t({{10, 12, 14}}));
    }

    WHEN("iterating over the grid")
    {
        auto begin = grid.begin();
        auto end = grid.end();

        CHECK_NE(begin, end);

        auto pos = 0;
        for (; begin != end; ++begin) {
            // I don't really want to repeat the logic completely again, but as long as we
            // in total traverse the same amount as size() of the grid, it should be fine
            ++pos;

            CAPTURE(*begin);
            CAPTURE(*end);

            // Protect against endless loops
            REQUIRE(pos < grid.size() + 1);
        }

        CHECK_EQ(pos, grid.size());
        CHECK_EQ(begin, end);
    }

    WHEN("Calculating distance")
    {
        auto begin = grid.begin();

        CHECK_EQ(begin - begin, 0);
        CHECK_EQ(std::distance(begin, begin), 0);

        auto iter = begin;
        ++iter;

        CHECK_EQ(begin - iter, -1);
        CHECK_EQ(std::distance(iter, begin), -1);

        CHECK_EQ(iter - begin, 1);
        CHECK_EQ(std::distance(begin, iter), 1);

        iter += grid.size() - 1;
        CHECK_EQ(iter - begin, grid.size());
    }
}

TEST_CASE("Visit neighbours in 2D")
{
    Eigen::IOFormat format(4, 0, ", ", "", "", "", "[", "]");

    auto cur = IndexVector_t({{2, 5}});
    auto lower = IndexVector_t({{2, 4}});
    auto upper = IndexVector_t({{3, 7}});
    // auto grid = CartesianIndices(lower, upper);
    auto grid = neighbours_in_slice(cur, 1);

    CHECK_EQ(lower, grid.first());
    CHECK_EQ(upper, grid.last());

    auto begin = grid.begin();
    auto end = grid.end();

    auto ypos = lower[1];
    for (; begin != end; ++begin) {
        CHECK_EQ(*begin, IndexVector_t({{cur[0], ypos}}));
        ++ypos;
    }
}

TEST_CASE("Visit neighbours in 2D")
{
    Eigen::IOFormat format(4, 0, ", ", "", "", "", "[", "]");

    auto cur = IndexVector_t({{2, 5}});
    auto lower = IndexVector_t({{2, 4}});
    auto upper = IndexVector_t({{3, 7}});
    // auto grid = CartesianIndices(lower, upper);
    auto grid = neighbours_in_slice(cur, 1);

    CHECK_EQ(lower, grid.first());
    CHECK_EQ(upper, grid.last());

    auto begin = grid.begin();
    auto end = grid.end();

    auto ypos = lower[1];
    for (; begin != end; ++begin) {
        CHECK_EQ(*begin, IndexVector_t({{cur[0], ypos}}));
        ++ypos;
    }
}

TEST_CASE("Visit neighbours in 2D with bounds")
{
    Eigen::IOFormat format(4, 0, ", ", "", "", "", "[", "]");
    WHEN("Having a start position at the lower border")
    {
        auto cur = IndexVector_t({{2, 1}});
        auto lower = IndexVector_t({{0, 0}});
        auto upper = IndexVector_t({{5, 5}});
        // auto grid = CartesianIndices(lower, upper);
        auto grid = neighbours_in_slice(cur, 2, lower, upper);

        CHECK_EQ(grid.first(), IndexVector_t({{2, 0}}));
        CHECK_EQ(grid.last(), IndexVector_t({{3, 4}}));

        auto begin = grid.begin();
        auto end = grid.end();

        auto ypos = 0;
        for (; begin != end; ++begin) {
            CHECK_EQ(*begin, IndexVector_t({{cur[0], ypos}}));
            ++ypos;
        }
    }

    WHEN("Having a start position at the upper border")
    {
        auto cur = IndexVector_t({{2, 4}});
        auto lower = IndexVector_t({{0, 0}});
        auto upper = IndexVector_t({{5, 5}});
        // auto grid = CartesianIndices(lower, upper);
        auto grid = neighbours_in_slice(cur, 2, lower, upper);

        CHECK_EQ(grid.first(), IndexVector_t({{2, 2}}));
        CHECK_EQ(grid.last(), IndexVector_t({{3, 5}}));

        auto begin = grid.begin();
        auto end = grid.end();

        auto ypos = 2;
        for (; begin != end; ++begin) {
            CHECK_EQ(*begin, IndexVector_t({{cur[0], ypos}}));
            ++ypos;
        }
    }
}

TEST_CASE("Test formatting")
{
    WHEN("Creating a 1D CartesianIndices")
    {
        const CartesianIndices grid(std::vector<int>{5});

        CHECK_EQ(fmt::format("{}", grid), "(0:5)");
    }

    WHEN("Creating a 1D CartesianIndices, not starting at 0")
    {
        const CartesianIndices grid(std::vector<int>{3}, std::vector<int>{5});

        CHECK_EQ(fmt::format("{}", grid), "(3:5)");
    }

    WHEN("Creating a 2D CartesianIndices")
    {
        const CartesianIndices grid(std::vector<int>{5, 3});

        CHECK_EQ(fmt::format("{}", grid), "(0:5, 0:3)");
    }

    WHEN("Creating a 1D CartesianIndices, not starting at 0")
    {
        const CartesianIndices grid(std::vector<int>{3, 1}, std::vector<int>{5, 5});

        CHECK_EQ(fmt::format("{}", grid), "(3:5, 1:5)");
    }

    WHEN("Creating a 3D CartesianIndices")
    {
        const CartesianIndices grid(std::vector<int>{5, 3, 9});

        CHECK_EQ(fmt::format("{}", grid), "(0:5, 0:3, 0:9)");
    }
}

TEST_CASE("Relational operators")
{
    GIVEN("a 1D grid")
    {
        const CartesianIndices grid(std::vector<int>{50});

        CAPTURE(grid);

        auto begin = grid.begin();
        auto end = grid.end();

        CAPTURE(*begin);
        CAPTURE(*end);

        WHEN("Taking some element in the middle of the grid")
        {
            auto mid = begin + 25;

            CAPTURE(*mid);

            THEN("It compare equal to begin + pos")
            {
                CHECK_EQ(mid, 25 + begin);
                CHECK_EQ(begin, mid - 25);
            }

            THEN("It behaves correctly for less then (equal)")
            {
                CHECK_LT(begin, mid);
                CHECK_LT(mid - 1, mid);
                CHECK_LE(mid - 1, mid);
                CHECK_LE(mid, mid);
            }

            THEN("It behaves correctly for less then (equal)")
            {
                CHECK_GT(mid, begin);
                CHECK_GT(1 + mid, mid);
                CHECK_GE(1 + mid, mid);
                CHECK_GE(mid, mid);
            }
        }
    }

    GIVEN("a 2D grid")
    {
        const CartesianIndices grid(std::vector<int>{10, 10});

        CAPTURE(grid);

        auto begin = grid.begin();
        auto end = grid.end();

        CAPTURE(*begin);
        CAPTURE(*end);

        WHEN("Taking some element in the middle of the grid")
        {
            auto mid = begin + 25;

            CAPTURE(*mid);

            THEN("It compare equal to begin + pos")
            {
                CHECK_EQ(mid, 25 + begin);
                CHECK_EQ(begin, mid - 25);
            }

            THEN("It behaves correctly for less then (equal)")
            {
                CHECK_LT(begin, mid);
                CHECK_LT(mid - 1, mid);
                CHECK_LE(mid - 1, mid);
                CHECK_LE(mid, mid);
            }

            THEN("It behaves correctly for less then (equal)")
            {
                CHECK_GT(mid, begin);
                CHECK_GT(1 + mid, mid);
                CHECK_GE(1 + mid, mid);
                CHECK_GE(mid, mid);
            }
        }
    }
}

TEST_SUITE_END();
