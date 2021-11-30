/**
 * @file test_DataHandlers.cpp
 *
 * @brief Common tests for CPU Handlers class
 *
 * @author David Frank - initial code
 */

#include "doctest/doctest.h"
#include "DataHandlerCPU.h"
#include "testHelpers.h"

using namespace elsa;
using namespace doctest;

template <typename data_t>
long useCount(const DataHandlerCPU<data_t>& dh)
{
    return dh.use_count();
}

TYPE_TO_STRING(DataHandlerCPU<float>);
TYPE_TO_STRING(DataHandlerCPU<double>);
TYPE_TO_STRING(DataHandlerCPU<index_t>);
TYPE_TO_STRING(DataHandlerCPU<std::complex<float>>);
TYPE_TO_STRING(DataHandlerCPU<std::complex<double>>);

TEST_SUITE_BEGIN("core");

// x = ...
// y = A(B(C(x)))

TEST_CASE_TEMPLATE("DataHandlerCPU: Test regularity", data_t, float, double, std::complex<float>,
                   std::complex<double>)
{
    constexpr index_t size = 42;

    GIVEN("An empty DataHandlerCPU")
    {
        DataHandlerCPU<data_t> x(size);

        THEN("It's size is correct") { CHECK_EQ(x.getSize(), size); }

        WHEN("Creating a copy")
        {
            DataHandlerCPU<data_t> y = x;

            THEN("Their sizes are equal") { CHECK_EQ(x.getSize(), y.getSize()); }

            THEN("They compare equally") { CHECK_EQ(x, y); }
        }

        WHEN("Moving it to a new handler")
        {
            DataHandlerCPU<data_t> y = std::move(x);

            THEN("The size is as the old one") { CHECK_EQ(y.getSize(), size); }
        }

        WHEN("Copy assigning one to another")
        {
            auto vec = generateRandomMatrix<data_t>(size * 2);
            DataHandlerCPU<data_t> y{vec};

            x = y;

            THEN("Their sizes are equal") { CHECK_EQ(x.getSize(), y.getSize()); }

            THEN("They are element wise equal") { CHECK_UNARY(isCwiseApprox(x, y)); }

            THEN("They equally compare") { CHECK_EQ(x, y); }
        }

        WHEN("Move assigning one to another")
        {
            auto vec = generateRandomMatrix<data_t>(size * 2);
            DataHandlerCPU<data_t> y{vec};

            x = std::move(y);

            THEN("the moved to has the size of the other") { CHECK_EQ(x.getSize(), vec.size()); }

            THEN("They are element wise equal") { CHECK_UNARY(isCwiseApprox(x, vec)); }
        }
    }

    GIVEN("A DataHandlerCPU")
    {
        auto vec = generateRandomMatrix<data_t>(size);

        DataHandlerCPU<data_t> x(vec);

        THEN("It's size is correct") { CHECK_EQ(x.getSize(), size); }

        WHEN("Creating a copy")
        {
            DataHandlerCPU<data_t> y = x;

            THEN("Their sizes are equal") { CHECK_EQ(x.getSize(), y.getSize()); }

            THEN("They are element wise equal") { CHECK_UNARY(isCwiseApprox(x, y)); }

            THEN("They equally compare") { CHECK_EQ(y, x); }
        }

        WHEN("Moving it to a new handler")
        {
            DataHandlerCPU<data_t> y = std::move(x);

            THEN("The size is as the old one") { CHECK_EQ(y.getSize(), size); }

            THEN("They are element wise equal") { CHECK_UNARY(isCwiseApprox(y, vec)); }
        }

        WHEN("Copy assigning one to another")
        {
            auto vec = generateRandomMatrix<data_t>(size * 2);
            DataHandlerCPU<data_t> y{vec};

            x = y;

            THEN("Their sizes are equal") { CHECK_EQ(x.getSize(), y.getSize()); }

            THEN("They are element wise equal") { CHECK_UNARY(isCwiseApprox(x, y)); }

            THEN("They equally compare") { CHECK_EQ(y, x); }
        }

        WHEN("Move assigning one to another")
        {
            auto vec = generateRandomMatrix<data_t>(size * 2);
            DataHandlerCPU<data_t> y{vec};

            x = std::move(y);

            THEN("the moved to has the size of the other") { CHECK_EQ(x.getSize(), vec.size()); }

            THEN("They are element wise equal") { CHECK_UNARY(isCwiseApprox(x, vec)); }
        }
    }
}

TEST_CASE_TEMPLATE("DataHandlerMapCPU: Test regularity", data_t, float, double, std::complex<float>,
                   std::complex<double>)
{
    constexpr index_t size = 42;

    auto v1 = generateRandomMatrix<data_t>(size);
    DataHandlerCPU<data_t> h1{v1};

    auto v2 = generateRandomMatrix<data_t>(size / 2);
    DataHandlerCPU<data_t> h2{v2};

    auto v3 = generateRandomMatrix<data_t>(size * 2);
    DataHandlerCPU<data_t> h3{v3};

    GIVEN("A map to a data handler")
    {
        auto x = DataHandlerMapCPU<data_t>(h1, 0, size);

        THEN("It's size is correct")
        {
            CHECK_EQ(x.getSize(), size);
            CHECK_EQ(x.getSize(), h1.getSize());
        }

        WHEN("Creating a copy")
        {
            DataHandlerMapCPU<data_t> y = x;

            THEN("Their sizes are equal") { CHECK_EQ(x.getSize(), y.getSize()); }

            THEN("They are element wise equal") { CHECK_UNARY(isCwiseApprox(x, y)); }

            THEN("They equally compare") { CHECK_EQ(x, y); }

            AND_WHEN("Writing to the original map changes the original handler")
            {
                x[0] = 100000;

                AND_THEN("The two maps are coefficient wise equal")
                {
                    CHECK_UNARY(isCwiseApprox(x, y));
                }

                AND_THEN("The two maps are coefficient wise equal to the map")
                {
                    CHECK_UNARY(isCwiseApprox(x, h1));
                    CHECK_UNARY(isCwiseApprox(y, h1));
                }
            }

            AND_WHEN("Writing to the original map changes the original handler")
            {
                y[0] = 100000;

                AND_THEN("The two maps are coefficient wise equal")
                {
                    CHECK_UNARY(isCwiseApprox(x, y));
                }

                AND_THEN("The two maps are coefficient wise equal to the map")
                {
                    CHECK_UNARY(isCwiseApprox(x, h1));
                    CHECK_UNARY(isCwiseApprox(y, h1));
                }
            }
        }

        WHEN("Moving it to a new handler")
        {
            DataHandlerMapCPU<data_t> y = std::move(x);

            THEN("The size is as the old one") { CHECK_EQ(y.getSize(), size); }

            THEN("They are element wise equal") { CHECK_UNARY(isCwiseApprox(y, v1)); }

            AND_WHEN("Writing to the new map changes the original handler")
            {
                y[0] = 100000;

                AND_THEN("The two maps are coefficient wise equal")
                {
                    CHECK_UNARY(isCwiseApprox(y, h1));
                }
            }
        }

        WHEN("Copy assigning a map to an smaller one")
        {
            DataHandlerMapCPU<data_t> y{h3, 0, h3.getSize()};

            x = y;

            THEN("Their sizes are equal")
            {
                CHECK_EQ(x.getSize(), y.getSize());
                CHECK_EQ(h3.getSize(), x.getSize());
                CHECK_NE(h1.getSize(), x.getSize());
            }

            THEN("They are element wise equal") { CHECK_UNARY(isCwiseApprox(x, y)); }

            THEN("They equally compare") { CHECK_EQ(y, x); }

            AND_WHEN("Writing to the original map")
            {
                x[0] = 100000;

                AND_THEN("The two maps are coefficient wise equal")
                {
                    CHECK_UNARY(isCwiseApprox(x, y));
                }

                AND_THEN("The two maps are coefficient wise equal to the map")
                {
                    CHECK_UNARY(isCwiseApprox(x, h3));
                    CHECK_UNARY(isCwiseApprox(y, h3));
                }
            }

            AND_WHEN("Writing to the assigning map")
            {
                y[0] = 100000;

                AND_THEN("The two maps are coefficient wise equal")
                {
                    CHECK_UNARY(isCwiseApprox(x, y));
                }

                AND_THEN("The two maps are coefficient wise equal to the map")
                {
                    CHECK_UNARY(isCwiseApprox(x, h3));
                    CHECK_UNARY(isCwiseApprox(y, h3));
                }
            }
        }

        WHEN("Copy assigning a map to an larger one")
        {
            DataHandlerMapCPU<data_t> y{h2, 0, h2.getSize()};

            x = y;

            THEN("Their sizes are equal")
            {
                CHECK_EQ(x.getSize(), y.getSize());
                CHECK_EQ(h2.getSize(), x.getSize());
                CHECK_NE(h1.getSize(), x.getSize());
            }

            THEN("They are element wise equal") { CHECK_UNARY(isCwiseApprox(x, y)); }

            THEN("They equally compare") { CHECK_EQ(y, x); }

            AND_WHEN("Writing to the original map")
            {
                x[0] = 100000;

                AND_THEN("The two maps are coefficient wise equal")
                {
                    CHECK_UNARY(isCwiseApprox(x, y));
                }

                AND_THEN("The two maps are coefficient wise equal to the map")
                {
                    CHECK_UNARY(isCwiseApprox(x, h2));
                    CHECK_UNARY(isCwiseApprox(y, h2));
                }
            }

            AND_WHEN("Writing to the assigning map")
            {
                y[0] = 100000;

                AND_THEN("The two maps are coefficient wise equal")
                {
                    CHECK_UNARY(isCwiseApprox(x, y));
                }

                AND_THEN("The two maps are coefficient wise equal to the map")
                {
                    CHECK_UNARY(isCwiseApprox(x, h2));
                    CHECK_UNARY(isCwiseApprox(y, h2));
                }
            }
        }

        WHEN("Move assigning a map to an smaller one")
        {
            DataHandlerMapCPU<data_t> y{h3, 0, h3.getSize()};

            x = std::move(y);

            THEN("The map has to same size as the moved from map had")
            {
                CHECK_EQ(h3.getSize(), x.getSize());
                CHECK_NE(h1.getSize(), x.getSize());
            }

            THEN("The map is coefficient wise equal to the underlying handler")
            {
                CHECK_UNARY(isCwiseApprox(x, h3));
            }

            AND_WHEN("Writing to the map")
            {
                x[0] = 100000;

                AND_THEN("the original handler is updated") { CHECK_UNARY(isCwiseApprox(x, h3)); }
            }
        }

        WHEN("Move assigning a map to an smaller one")
        {
            DataHandlerMapCPU<data_t> y{h2, 0, h2.getSize()};

            x = std::move(y);

            THEN("The map has to same size as the moved from map had")
            {
                CHECK_EQ(h2.getSize(), x.getSize());
                CHECK_NE(h1.getSize(), x.getSize());
            }

            THEN("The map is coefficient wise equal to the underlying handler")
            {
                CHECK_UNARY(isCwiseApprox(x, h2));
            }

            AND_WHEN("Writing to the map")
            {
                x[0] = 100000;

                AND_THEN("the original handler is updated") { CHECK_UNARY(isCwiseApprox(x, h2)); }
            }
        }
    }
}

TEST_CASE_TEMPLATE("CPUHandlers: Testing equality operator", data_t, float, double,
                   std::complex<float>, std::complex<double>)
{
    constexpr index_t size = 42;

    const auto v1 = generateRandomMatrix<data_t>(size);
    const auto v2 = generateRandomMatrix<data_t>(size);

    GIVEN("Just a handler")
    {
        const DataHandlerCPU<data_t> x{v1};

        THEN("It compare equally to itself") { CHECK_EQ(x, x); }
    }

    GIVEN("Two CPU handlers from the same vectors")
    {
        DataHandlerCPU<data_t> x{v1};
        DataHandlerCPU<data_t> y{v1};

        THEN("They compare equally") { CHECK_EQ(x, y); }

        AND_WHEN("Writing to one of the CPU handlers")
        {
            x[0] = 123;

            THEN("They don't compare equally anymore") { CHECK_NE(x, y); }
        }

        AND_WHEN("Writing to one of the CPU handlers")
        {
            y[0] = 123;

            THEN("They don't compare equally anymore") { CHECK_NE(x, y); }
        }
    }

    GIVEN("Two CPU handlers from the different vectors")
    {
        const DataHandlerCPU<data_t> x{v1};
        const DataHandlerCPU<data_t> y{v2};

        THEN("They don't compare equally") { CHECK_NE(x, y); }
    }

    GIVEN("A CPU handler and a map to the complete CPU handler")
    {
        DataHandlerCPU<data_t> x{v1};
        DataHandlerMapCPU<data_t> m{x, 0, size};

        THEN("The map compare equally to itself") { CHECK_EQ(m, m); }

        THEN("They compare equally")
        {
            CHECK_EQ(x, m);
            CHECK_EQ(m, x);
        }

        AND_WHEN("Writing to the owning handler")
        {
            x[0] *= 123;
            THEN("They still compare equally")
            {
                CHECK_EQ(x, m);
                CHECK_EQ(m, x);
            }
        }

        AND_WHEN("Writing to the map handler")
        {
            m[0] *= 123;
            THEN("They still compare equally")
            {
                CHECK_EQ(x, m);
                CHECK_EQ(m, x);
            }
        }
    }

    GIVEN("An owning CPU handler and a map to a portion of the owning handler")
    {
        DataHandlerCPU<data_t> x{v1};
        DataHandlerMapCPU<data_t> m{x, 10, size / 2};

        THEN("The map compare equally to itself") { CHECK_EQ(m, m); }

        THEN("They don't have the same size") { CHECK_NE(x.getSize(), m.getSize()); }

        THEN("They compare equally")
        {
            CHECK_NE(x, m);
            CHECK_NE(m, x);
        }

        AND_WHEN("Having a new map of the same size, to a different part")
        {
            DataHandlerMapCPU<data_t> n{x, 0, size / 2};

            AND_THEN("The maps don't compare equally") { CHECK_NE(m, n); }
        }
    }

    GIVEN("Two CPU handlers and with a map each to the whole data handler")
    {
        DataHandlerCPU<data_t> x{v1};
        DataHandlerMapCPU<data_t> m{x, 0, size};

        DataHandlerCPU<data_t> y{v2};
        DataHandlerMapCPU<data_t> n{y, 0, size};

        THEN("The two maps don't compare equally") { CHECK_NE(n, m); }

        THEN("The map doesn't compare equally to the other data handler")
        {
            CHECK_NE(x, n);
            CHECK_NE(n, x);

            CHECK_NE(y, m);
            CHECK_NE(m, y);
        }

        AND_WHEN("Copy assigning the data containers")
        {
            x = y;

            AND_THEN("They compare equally") { CHECK_EQ(x, y); }
            AND_THEN("The maps compare equally") { CHECK_EQ(n, m); }
        }
    }
}

TEST_CASE_TEMPLATE("CPUHandlers: Testing isEqual", data_t, float, double, std::complex<float>,
                   std::complex<double>)
{
    constexpr index_t size = 42;

    const auto v1 = generateRandomMatrix<data_t>(size);
    const auto v2 = generateRandomMatrix<data_t>(2 * size);

    GIVEN("An owning CPU handler and a block of it covering the whole handler")
    {
        DataHandlerCPU<data_t> x{v1};
        auto block = x.getBlock(0, size);

        THEN("They have the same size") { CHECK_EQ(x.getSize(), block->getSize()); }

        THEN("They compare equally")
        {
            CHECK_EQ(x, *block);
            CHECK_EQ(*block, x);
        }

        AND_WHEN("Having a new map of the same size, to a different part")
        {
            auto smallBlock = x.getBlock(0, size / 2);

            AND_THEN("They don't compare equally")
            {
                CHECK_NE(x, *smallBlock);
                CHECK_NE(*smallBlock, x);
            }
        }
    }

    GIVEN("An owning CPU handler and a map of compatible size to a larger CPU handler")
    {
        DataHandlerCPU<data_t> x{v1};
        DataHandlerCPU<data_t> y{v2};
        auto block = y.getBlock(0, size);
        auto& b = *block;

        THEN("They have the same size") { CHECK_EQ(x.getSize(), b.getSize()); }

        THEN("They don't compare equally")
        {
            CHECK_NE(x, b);
            CHECK_NE(b, x);
        }

        AND_WHEN("Assigning the handler to the block")
        {
            b = x;

            // TODO:
            // AND_THEN("They do what?")
            // {
            //     CHECK_EQ(b, x);
            //     CHECK_EQ(x, b);
            // }
        }
    }
}

TEST_CASE_TEMPLATE("DataHandlerCPU: Testing polymorphic assignment", data_t, float, double,
                   std::complex<float>, std::complex<double>)
{
    constexpr index_t size = 42;

    const auto v1 = generateRandomMatrix<data_t>(size);
    const auto v2 = generateRandomMatrix<data_t>(2 * size);

    GIVEN("An owning CPU handler and a map to another of compatible size")
    {
        DataHandlerCPU<data_t> x{v1};
        DataHandlerCPU<data_t> y{v2};
        auto block = y.getBlock(0, size);

        WHEN("Assigning the handler to the map")
        {
            *block = x;

            // TODO:
            // THEN("They compare equally")
            // {
            //     CHECK_EQ(*block, x);
            //     CHECK_EQ(x, *block);
            // }
        }
    }
}

TEST_CASE_TEMPLATE("DataHandlerCPU: Testing cloning", data_t, float, double, std::complex<float>,
                   std::complex<double>)
{
    constexpr index_t size = 42;

    const auto v1 = generateRandomMatrix<data_t>(size);
    const auto v2 = generateRandomMatrix<data_t>(2 * size);

    GIVEN("Given a DataHandlerCPU")
    {
        DataHandlerCPU<data_t> x{v1};

        WHEN("Creating a clone if it")
        {
            auto clone = x.clone();

            THEN("The clone compare equally to the original handler")
            {
                CHECK_EQ(*clone, x);
                CHECK_EQ(x, *clone);
            }

            THEN("The clone is coefficient wise equal to the original handler")
            {
                CHECK_UNARY(isCwiseApprox(*clone, x));
            }
        }
    }

    GIVEN("Given a DataHandlerMapCPU")
    {
        DataHandlerCPU<data_t> h{v1};
        DataHandlerMapCPU<data_t> x{h, 0, size};

        WHEN("Creating a clone if it")
        {
            auto clone = x.clone();

            THEN("The clone compare equally to the original handler")
            {
                CHECK_EQ(*clone, x);
                CHECK_EQ(x, *clone);
            }

            THEN("The clone is coefficient wise equal to the original handler")
            {
                CHECK_UNARY(isCwiseApprox(*clone, x));
            }

            AND_WHEN("Writing to the clone")
            {
                (*clone)[0] += 123;

                THEN("The clone is still coefficient wise equal to the original handler")
                {
                    CHECK_UNARY(isCwiseApprox(*clone, x));
                }
            }

            AND_WHEN("Writing to the original handler")
            {
                (*clone)[0] += 123;

                THEN("The clone is still coefficient wise equal to the original handler")
                {
                    CHECK_UNARY(isCwiseApprox(*clone, x));
                }
            }
        }
    }
}

TEST_CASE_TEMPLATE("DataHandlerCPU: Testing reduction operations", data_t, float, double,
                   std::complex<float>, std::complex<double>)
{
    constexpr auto eps = std::numeric_limits<GetFloatingPointType_t<data_t>>::epsilon();
    constexpr index_t size = 16;

    auto v = generateRandomMatrix<data_t>(size);

    GIVEN("An owning CPU Handler")
    {
        DataHandlerCPU<data_t> x{v};

        THEN("The reductions and norms yield the expected output")
        {
            CHECK_UNARY(checkApproxEq(x.sum(), v.sum()));
            CHECK_UNARY(checkApproxEq(x.l0PseudoNorm(), (v.array().cwiseAbs() >= eps).count()));
            CHECK_UNARY(checkApproxEq(x.l1Norm(), v.array().abs().sum()));
            CHECK_UNARY(checkApproxEq(x.l2Norm(), v.norm()));
            CHECK_UNARY(checkApproxEq(x.squaredL2Norm(), v.squaredNorm()));
            CHECK_UNARY(checkApproxEq(x.lInfNorm(), v.array().abs().maxCoeff()));
        }

        THEN("The dot product with itself is correct")
        {
            CHECK_UNARY(checkApproxEq(x.dot(x), v.dot(v)));
            CHECK_UNARY(checkApproxEq(x.dot(x), x.squaredL2Norm()));
        }

        AND_WHEN("there is a second owning CPU handler")
        {
            auto w = generateRandomMatrix<data_t>(size);
            DataHandlerCPU<data_t> y{w};

            AND_THEN("The dot product is correct")
            {
                CHECK_UNARY(checkApproxEq(x.dot(y), v.dot(w)));
                CHECK_UNARY(checkApproxEq(y.dot(x), w.dot(v)));
            }
        }
    }
}

TEST_CASE_TEMPLATE("CPUHandlers: Testing the element-wise operations", data_t, float, double,
                   std::complex<float>, std::complex<double>)
{
    constexpr index_t size = 42;

    auto v = generateRandomMatrix<data_t>(size);
    auto w = generateRandomMatrix<data_t>(size);

    auto eval = [](auto&& arg) {
        using T = decltype(arg);
        return Vector_t<data_t>(std::forward<T>(arg));
    };

    GIVEN("Two owning data handlers of the same size")
    {
        DataHandlerCPU<data_t> x{v};
        DataHandlerCPU<data_t> y{w};

        WHEN("Computing x += y")
        {
            x += y;

            CHECK_UNARY(isCwiseApprox(x, eval(v + w)));
        }

        WHEN("Computing x -= y")
        {
            x -= y;

            CHECK_UNARY(isCwiseApprox(x, eval(v - w)));
        }

        WHEN("Computing x *= y")
        {
            x *= y;

            CHECK_UNARY(isCwiseApprox(x, eval(v.array() * w.array())));
        }

        WHEN("Computing x /= y")
        {
            x /= y;

            CHECK_UNARY(isCwiseApprox(x, eval(v.array() / w.array())));
        }

        WHEN("given a scalar s")
        {
            const auto a = [] {
                if constexpr (isComplex<data_t>) {
                    return data_t{4, 2};
                } else {
                    return data_t{42};
                }
            }();

            AND_WHEN("computing x += a")
            {
                x += a;

                CHECK_UNARY(isCwiseApprox(x, eval(v.array() + a)));
            }

            AND_WHEN("computing x -= a")
            {
                x -= a;

                CHECK_UNARY(isCwiseApprox(x, eval(v.array() - a)));
            }

            AND_WHEN("computing x *= a")
            {
                x *= a;

                CHECK_UNARY(isCwiseApprox(x, eval(v.array() * a)));
            }

            AND_WHEN("computing x /= a")
            {
                x /= a;

                CHECK_UNARY(isCwiseApprox(x, eval(v.array() / a)));
            }

            AND_WHEN("computing x = a")
            {
                x = a;

                CHECK_UNARY(isCwiseApprox(x, eval(v.array() = a)));
            }
        }
    }
}

TEST_SUITE_END();
