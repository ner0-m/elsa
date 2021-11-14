
#include "doctest/doctest.h"
#include "Into.hpp"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("core");

struct ConvertibleToInt {
    operator int() { return 42; }
};

struct NotConvertibleToInt {
};

int foo(Into<int> into)
{
    return into.into();
}

std::optional<int> tryfoo(TryInto<int> into)
{
    return into.tryInto();
}

TEST_CASE("Into: Call function which takes Into<int> instead of int")
{
    WHEN("Calling with a convertible class")
    {
        THEN("The converted value is returned") { CHECK_EQ(foo(ConvertibleToInt{}), 42); }
    }

    WHEN("Calling with T (int in this case) itself")
    {
        THEN("The value is returned") { CHECK_EQ(foo(42), 42); }
    }
}

TEST_CASE("TryInto: Calling function taking TryInto<int> as argument")
{
    WHEN("Calling with a convertible class")
    {
        THEN("The converted value is returned") { CHECK_EQ(*tryfoo(ConvertibleToInt{}), 42); }
    }

    WHEN("Calling with T (int in this case) itself")
    {
        THEN("The value is returned") { CHECK_EQ(*tryfoo(42), 42); }
    }

    WHEN("Calling with a non convertible class")
    {
        THEN("nulltopt is returned") { CHECK_EQ(tryfoo(NotConvertibleToInt{}), std::nullopt); }
    }
}

TEST_SUITE_END();
