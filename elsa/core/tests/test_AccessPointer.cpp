#include "doctest/doctest.h"
#include "AccessPointer.hpp"

using namespace elsa;

TEST_SUITE_BEGIN("core");

TEST_CASE("CopyOnWritePointer")
{
    GIVEN("A CopyOnWritePointer to an int")
    {
        CopyOnWritePointer<int> x = 42;

        THEN("It's unique")
        {
            CHECK_UNARY(x.unique());
            CHECK_EQ(x.use_count(), 1);
        }

        THEN("Itself is the identity") { CHECK_UNARY(x.identity(x)); }

        THEN("The stored value is correct")
        {
            CHECK_EQ(x.read(), 42);
            CHECK_EQ(*x, 42);
        }

        THEN("It's convertible to true") { CHECK_UNARY(static_cast<bool>(x)); }

        THEN("Comparison operators with element type works as expected yield desired result")
        {
            // Test all combinations and have element type as first and second argument
            CHECK_EQ(x, 42);
            CHECK_EQ(42, x);

            CHECK_NE(x, 69);
            CHECK_NE(69, x);

            CHECK_GT(x, 41);
            CHECK_LT(41, x);

            CHECK_GE(x, 41);
            CHECK_LE(41, x);

            CHECK_GE(x, 42);
            CHECK_GE(42, x);

            CHECK_LT(x, 43);
            CHECK_GT(43, x);

            CHECK_LE(x, 43);
            CHECK_GE(43, x);

            CHECK_LE(x, 42);
            CHECK_LE(42, x);
        }

        WHEN("Assigning a new value to the ptr")
        {
            x = 69;
            THEN("The stored value is updated")
            {
                CHECK_EQ(x.read(), 69);
                CHECK_EQ(*x, 69);
            }
        }

        WHEN("Having a second CopyOnWritePointer , with the same value by change")
        {
            CopyOnWritePointer<int> y = 42;

            THEN("They compare equally") { CHECK_EQ(x, y); }
            THEN("They don't have the same identity") { CHECK_UNARY_FALSE(x.identity(y)); }
        }

        WHEN("Having a second CopyOnWritePointer , with any value")
        {
            CopyOnWritePointer<int> y = 69;

            THEN("They don't compare equally") { CHECK_NE(x, y); }

            THEN("They don't have the same identity") { CHECK_UNARY_FALSE(x.identity(y)); }

            AND_WHEN("Copy Assigning y to x")
            {
                x = y;

                THEN("They are not unique anymore")
                {
                    CHECK_UNARY_FALSE(y.unique());
                    CHECK_UNARY_FALSE(x.unique());
                }

                THEN("Both have use_count() of 2")
                {
                    CHECK_EQ(y.use_count(), 2);
                    CHECK_EQ(x.use_count(), 2);
                }

                THEN("They have the same identity") { CHECK_UNARY(x.identity(y)); }

                THEN("They compare equally") { CHECK_EQ(x, y); }

                THEN("Both have the same value")
                {
                    CHECK_EQ(*x, *y);
                    CHECK_EQ(x.read(), y.read());
                }

                AND_WHEN("Writing to the new pointer")
                {
                    y = 42;

                    THEN("They are unique again")
                    {
                        CHECK_UNARY(y.unique());
                        CHECK_UNARY(x.unique());
                    }

                    THEN("Both have use_count() of 1 again")
                    {
                        CHECK_EQ(y.use_count(), 1);
                        CHECK_EQ(x.use_count(), 1);
                    }

                    THEN("They don't have the same identity") { CHECK_UNARY_FALSE(x.identity(y)); }

                    THEN("The values are updated in the copy")
                    {
                        CHECK_EQ(*y, 42);
                        CHECK_EQ(y.read(), 42);
                    }

                    THEN("The original pointer's value is untouched")
                    {
                        CHECK_EQ(*x, 69);
                        CHECK_EQ(x.read(), 69);
                    }
                }
            }
        }

        WHEN("Creating a copy of the ptr (via copy constructor)")
        {
            CopyOnWritePointer<int> y = x;

            THEN("Both are not unique")
            {
                CHECK_UNARY_FALSE(y.unique());
                CHECK_UNARY_FALSE(x.unique());
            }

            THEN("One is the identity of the other") { CHECK_UNARY(x.identity(y)); }

            THEN("Both have use_count() of 2")
            {
                CHECK_EQ(y.use_count(), 2);
                CHECK_EQ(x.use_count(), 2);
            }

            THEN("Both have the same value")
            {
                CHECK_EQ(*x, *y);
                CHECK_EQ(x.read(), y.read());
            }

            AND_WHEN("Writing to the copy")
            {
                y = 69;

                THEN("They are unique again")
                {
                    CHECK_UNARY(y.unique());
                    CHECK_UNARY(x.unique());
                }

                THEN("Both have use_count() of 1 again")
                {
                    CHECK_EQ(y.use_count(), 1);
                    CHECK_EQ(x.use_count(), 1);
                }

                THEN("They don't have the same identity") { CHECK_UNARY_FALSE(x.identity(y)); }

                THEN("The values are updated in the copy")
                {
                    CHECK_EQ(*y, 69);
                    CHECK_EQ(y.read(), 69);
                }

                THEN("The original pointer's value is untouched")
                {
                    CHECK_EQ(*x, 42);
                    CHECK_EQ(x.read(), 42);
                }
            }
        }
    }
}

TEST_CASE("ObserverPointer")
{
    GIVEN("A ObserverPointer to an int")
    {
        ObserverPointer<int> x = 42;

        THEN("It's unique")
        {
            CHECK_UNARY(x.unique());
            CHECK_EQ(x.use_count(), 1);
        }

        THEN("Itself is the identity") { CHECK_UNARY(x.identity(x)); }

        THEN("The stored value is correct")
        {
            CHECK_EQ(x.read(), 42);
            CHECK_EQ(*x, 42);
        }

        THEN("It's convertible to true") { CHECK_UNARY(static_cast<bool>(x)); }

        THEN("Comparison operators with element type works as expected yield desired result")
        {
            // Test all combinations and have element type as first and second argument
            CHECK_EQ(x, 42);
            CHECK_EQ(42, x);

            CHECK_NE(x, 69);
            CHECK_NE(69, x);

            CHECK_GT(x, 41);
            CHECK_LT(41, x);

            CHECK_GE(x, 41);
            CHECK_LE(41, x);

            CHECK_GE(x, 42);
            CHECK_GE(42, x);

            CHECK_LT(x, 43);
            CHECK_GT(43, x);

            CHECK_LE(x, 43);
            CHECK_GE(43, x);

            CHECK_LE(x, 42);
            CHECK_LE(42, x);
        }

        WHEN("Assigning a new value to the ptr")
        {
            x = 69;
            THEN("The stored value is updated")
            {
                CHECK_EQ(x.read(), 69);
                CHECK_EQ(*x, 69);
            }
        }

        WHEN("Having a second ObserverPointer pointer, with the same value by change")
        {
            ObserverPointer<int> y = 42;

            THEN("They compare equally")
            {
                CHECK_EQ(x, y);
                CHECK_EQ(*x, *y);
                CHECK_EQ(x.read(), y.read());
            }

            THEN("They don't have the same identity") { CHECK_UNARY_FALSE(x.identity(y)); }
        }

        WHEN("Having a second ObserverPointer pointer, with any value")
        {
            ObserverPointer<int> y = 69;

            THEN("They don't compare equally") { CHECK_NE(x, y); }

            THEN("They don't have the same identity") { CHECK_UNARY_FALSE(x.identity(y)); }

            AND_WHEN("Copy Assigning y to x")
            {
                x = y;

                THEN("They share the same identity") { CHECK_UNARY(x.identity(x)); }

                THEN("They are not unique anymore")
                {
                    CHECK_UNARY_FALSE(y.unique());
                    CHECK_UNARY_FALSE(x.unique());
                }

                THEN("Both have use_count() of 2")
                {
                    CHECK_EQ(y.use_count(), 2);
                    CHECK_EQ(x.use_count(), 2);
                }

                THEN("Both have the same value")
                {
                    CHECK_EQ(*x, *y);
                    CHECK_EQ(x.read(), y.read());
                }

                AND_WHEN("Writing to the new pointer")
                {
                    y = 42;

                    THEN("They still have the same identity") { CHECK_UNARY(x.identity(x)); }

                    THEN("They are still not unique")
                    {
                        CHECK_UNARY_FALSE(y.unique());
                        CHECK_UNARY_FALSE(x.unique());
                    }

                    THEN("Both still have a use_count() of 2")
                    {
                        CHECK_EQ(y.use_count(), 2);
                        CHECK_EQ(x.use_count(), 2);
                    }

                    THEN("They still compare equally") { CHECK_EQ(x, y); }

                    THEN("The values are updated in the copy")
                    {
                        CHECK_EQ(*y, 42);
                        CHECK_EQ(y.read(), 42);
                    }

                    THEN("The original pointer's value is updated")
                    {
                        CHECK_EQ(*x, 42);
                        CHECK_EQ(x.read(), 42);
                    }
                }
            }
        }

        WHEN("Creating a copy of the ptr (via copy constructor)")
        {
            ObserverPointer<int> y = x;

            THEN("Both are not unique")
            {
                CHECK_UNARY_FALSE(y.unique());
                CHECK_UNARY_FALSE(x.unique());
            }

            THEN("One is the identity of the other") { CHECK_UNARY(x.identity(y)); }

            THEN("Both have use_count() of 2")
            {
                CHECK_EQ(y.use_count(), 2);
                CHECK_EQ(x.use_count(), 2);
            }

            THEN("They compare equally") { CHECK_EQ(x, y); }

            THEN("Both have the same value")
            {
                CHECK_EQ(*x, *y);
                CHECK_EQ(x.read(), y.read());
            }

            AND_WHEN("Writing to the copy")
            {
                y = 69;

                THEN("They are still not unique")
                {
                    CHECK_UNARY_FALSE(y.unique());
                    CHECK_UNARY_FALSE(x.unique());
                }

                THEN("Both still have a use_count() of 2")
                {
                    CHECK_EQ(y.use_count(), 2);
                    CHECK_EQ(x.use_count(), 2);
                }

                THEN("They still compare equally") { CHECK_EQ(x, y); }

                THEN("The values are updated in the copy")
                {
                    CHECK_EQ(*y, 69);
                    CHECK_EQ(y.read(), 69);
                }

                THEN("The original pointer's value is updated")
                {
                    CHECK_EQ(*x, 69);
                    CHECK_EQ(x.read(), 69);
                }
            }
        }
    }
}

TEST_SUITE_END();
