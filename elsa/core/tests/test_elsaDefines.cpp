/**
 * @file test_elsaDefines.cpp
 *
 * @brief Tests for common elsa defines
 *
 * @author David Frank - initial version
 */

#include "doctest/doctest.h"
#include <iostream>
#include "elsaDefines.h"
#include "TypeCasts.hpp"

#include <type_traits>

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("core");

TEST_CASE("elsaDefines: Testing PI")
{

    THEN("Pi for real_t and pi_t are equal") { REQUIRE_EQ(pi<real_t>, pi_t); }

    THEN("pi_t is somewhat close to a representation for pi")
    {
        REQUIRE_EQ(pi_t, Approx(3.14159265358979323846).epsilon(1e-5));
    }

    THEN("Pi for double is close to given value for pi")
    {
        REQUIRE_EQ(pi<double>, 3.14159265358979323846);
    }
}

TEST_CASE("elsaDefines: Testing compile-time predicates")
{
    static_assert(std::is_same_v<float, GetFloatingPointType_t<std::complex<float>>>);
    static_assert(std::is_same_v<double, GetFloatingPointType_t<std::complex<double>>>);
    static_assert(std::is_same_v<double, GetFloatingPointType_t<double>>);
    static_assert(std::is_same_v<float, GetFloatingPointType_t<float>>);
    static_assert(!std::is_same_v<float, GetFloatingPointType_t<double>>);

    REQUIRE_UNARY(true);
}

TEST_CASE("elsaDefines: Printing default handler type")
{
#ifdef ELSA_CUDA_VECTOR
    REQUIRE(defaultHandlerType == DataHandlerType::GPU);
#else
    REQUIRE(defaultHandlerType == DataHandlerType::CPU);
#endif
}

TEST_CASE("TypeCasts: Casting to unsigned")
{
    WHEN("Value is in range of the destination type")
    {
        std::int8_t signed1{100};
        std::int16_t signed2{100};
        std::int32_t signed3{100};
        std::int64_t signed4{100};

        auto unsigned1 = asUnsigned(signed1);
        auto unsigned2 = asUnsigned(signed2);
        auto unsigned3 = asUnsigned(signed3);
        auto unsigned4 = asUnsigned(signed4);

        static_assert(std::is_same_v<decltype(unsigned1), std::make_unsigned_t<decltype(signed1)>>);
        static_assert(std::is_same_v<decltype(unsigned2), std::make_unsigned_t<decltype(signed2)>>);
        static_assert(std::is_same_v<decltype(unsigned3), std::make_unsigned_t<decltype(signed3)>>);
        static_assert(std::is_same_v<decltype(unsigned4), std::make_unsigned_t<decltype(signed4)>>);

        REQUIRE_EQ(signed1, unsigned1);
        REQUIRE_EQ(signed2, unsigned2);
        REQUIRE_EQ(signed3, unsigned3);
        REQUIRE_EQ(signed4, unsigned4);
    }

    WHEN("Value is already unsigned")
    {
        std::uint8_t val1{100};
        std::uint16_t val2{100};
        std::uint32_t val3{100};
        std::uint64_t val4{100};

        auto unsigned1 = asUnsigned(val1);
        auto unsigned2 = asUnsigned(val2);
        auto unsigned3 = asUnsigned(val3);
        auto unsigned4 = asUnsigned(val4);

        static_assert(std::is_same_v<decltype(unsigned1), decltype(val1)>);
        static_assert(std::is_same_v<decltype(unsigned2), decltype(val2)>);
        static_assert(std::is_same_v<decltype(unsigned3), decltype(val3)>);
        static_assert(std::is_same_v<decltype(unsigned4), decltype(val4)>);

        REQUIRE_EQ(val1, unsigned1);
        REQUIRE_EQ(val2, unsigned2);
        REQUIRE_EQ(val3, unsigned3);
        REQUIRE_EQ(val4, unsigned4);
    }
}

TEST_CASE("TypeCasts: Casting to signed")
{
    WHEN("Value is in range of the destination type")
    {
        std::uint8_t unsigned1{100};
        std::uint16_t unsigned2{100};
        std::uint32_t unsigned3{100};
        std::uint64_t unsigned4{100};

        auto signed1 = asSigned(unsigned1);
        auto signed2 = asSigned(unsigned2);
        auto signed3 = asSigned(unsigned3);
        auto signed4 = asSigned(unsigned4);

        static_assert(std::is_same_v<decltype(signed1), std::make_signed_t<decltype(unsigned1)>>);
        static_assert(std::is_same_v<decltype(signed2), std::make_signed_t<decltype(unsigned2)>>);
        static_assert(std::is_same_v<decltype(signed3), std::make_signed_t<decltype(unsigned3)>>);
        static_assert(std::is_same_v<decltype(signed4), std::make_signed_t<decltype(unsigned4)>>);

        REQUIRE_EQ(signed1, unsigned1);
        REQUIRE_EQ(signed2, unsigned2);
        REQUIRE_EQ(signed3, unsigned3);
        REQUIRE_EQ(signed4, unsigned4);
    }

    WHEN("Value is already signed")
    {
        std::int8_t val1{100};
        std::int16_t val2{100};
        std::int32_t val3{100};
        std::int64_t val4{100};

        auto signed1 = asSigned(val1);
        auto signed2 = asSigned(val2);
        auto signed3 = asSigned(val3);
        auto signed4 = asSigned(val4);

        static_assert(std::is_same_v<decltype(signed1), decltype(val1)>);
        static_assert(std::is_same_v<decltype(signed2), decltype(val2)>);
        static_assert(std::is_same_v<decltype(signed3), decltype(val3)>);
        static_assert(std::is_same_v<decltype(signed4), decltype(val4)>);

        REQUIRE_EQ(val1, signed1);
        REQUIRE_EQ(val2, signed2);
        REQUIRE_EQ(val3, signed3);
        REQUIRE_EQ(val4, signed4);
    }
}

TEST_CASE("TypeCasts: Testing is()")
{
    struct Base {
        virtual ~Base() = default;
    };
    struct Derived1 final : Base {
        ~Derived1() override = default;
    };
    struct Derived2 final : public Base {
        ~Derived2() override = default;
    };

    WHEN("Base pointer points to Derived1")
    {
        std::unique_ptr<Base> ptr = std::make_unique<Derived1>();

        THEN("Casting to base is also fine")
        {
            REQUIRE_UNARY(is<Base>(ptr.get()));
            REQUIRE_UNARY(ptr);
        }
        THEN("Casting to Derived1 is fine")
        {
            REQUIRE_UNARY(is<Derived1>(ptr.get()));
            REQUIRE_UNARY(ptr);
        }
        THEN("Casting to Derived2 doesn't work")
        {
            REQUIRE_UNARY_FALSE(is<Derived2>(ptr.get()));
            REQUIRE_UNARY(ptr);
        }
    }

    WHEN("Base reference points to Derived1")
    {
        std::unique_ptr<Base> ptr = std::make_unique<Derived1>();

        THEN("Casting to base is also fine")
        {
            REQUIRE_UNARY(is<Base>(*ptr));
            REQUIRE_UNARY(ptr);
        }
        THEN("Casting to Derived1 is fine")
        {
            REQUIRE_UNARY(is<Derived1>(*ptr));
            REQUIRE_UNARY(ptr);
        }
        THEN("Casting to Derived2 doesn't work")
        {
            REQUIRE_UNARY_FALSE(is<Derived2>(*ptr));
            REQUIRE_UNARY(ptr);
        }
    }

    WHEN("unique_ptr to Base points to Derived1")
    {
        std::unique_ptr<Base> ptr = std::make_unique<Derived1>();

        THEN("Check is<Base> returns true")
        {
            REQUIRE_UNARY(is<Base>(ptr));
            REQUIRE_UNARY(ptr);
        }
        THEN("Check is<Derived1> returns true")
        {
            REQUIRE_UNARY(is<Derived1>(ptr));
            REQUIRE_UNARY(ptr);
        }
        THEN("Check is<Derived2> returns false")
        {
            REQUIRE_UNARY_FALSE(is<Derived2>(ptr));
            REQUIRE_UNARY(ptr);
        }
    }
}

TEST_CASE("TypeCasts: Testing downcast")
{
    struct Base {
        virtual ~Base() = default;
    };
    struct Derived1 final : Base {
        ~Derived1() override = default;
    };
    struct Derived2 final : public Base {
        ~Derived2() override = default;
    };

    WHEN("Base pointer points to Derived1")
    {
        std::unique_ptr<Base> ptr = std::make_unique<Derived1>();

        THEN("Downcasting to Base works")
        {
            REQUIRE_UNARY(downcast<Base>(ptr.get()));
            REQUIRE_UNARY(ptr);
        }
        THEN("Downcasting to Derived1 works")
        {
            REQUIRE_UNARY(downcast<Derived1>(ptr.get()));
            REQUIRE_UNARY(ptr);
        }
        THEN("Safely downcasting to Derived1 works")
        {
            REQUIRE_UNARY(downcast_safe<Derived1>(ptr.get()));
            REQUIRE_UNARY(ptr);
        }
        THEN("Safely downcasting to Derived2 doesn't work")
        {
            REQUIRE_UNARY_FALSE(downcast_safe<Derived2>(ptr.get()));
            REQUIRE_UNARY(ptr);
        }
    }

    WHEN("Base reference points to Derived1")
    {
        std::unique_ptr<Base> ptr = std::make_unique<Derived1>();

        THEN("Downcasting to Base works")
        {
            REQUIRE_NOTHROW(downcast<Base>(*ptr));
            REQUIRE_UNARY(ptr);
        }
        THEN("Downcasting to Derived1 works")
        {
            REQUIRE_NOTHROW(downcast<Derived1>(*ptr));
            REQUIRE_UNARY(ptr);
        }
        THEN("Safely downcasting to Derived1 works")
        {
            REQUIRE_NOTHROW(downcast_safe<Derived1>(*ptr));
            REQUIRE_UNARY(ptr);
        }
        THEN("Safely downcasting to Derived2 throws")
        {
            REQUIRE_THROWS(downcast_safe<Derived2>(*ptr));
            REQUIRE_UNARY(ptr);
        }
    }

    WHEN("unique_ptr to Base points to Derived1")
    {
        std::unique_ptr<Base> ptr = std::make_unique<Derived1>();

        THEN("Downcasting to base works")
        {
            REQUIRE_UNARY(downcast<Base>(std::move(ptr)));
            REQUIRE_UNARY_FALSE(ptr);
        }
        THEN("Downcasting to Derived1 works")
        {
            REQUIRE_UNARY(downcast<Derived1>(std::move(ptr)));
            REQUIRE_UNARY_FALSE(ptr);
        }
        THEN("Safely downcasting to Derived1 works")
        {
            REQUIRE_UNARY(downcast_safe<Derived1>(std::move(ptr)));
            REQUIRE_UNARY_FALSE(ptr);
        }
        THEN("Safely downcasting to Derived2 doesn't work")
        {
            REQUIRE_UNARY_FALSE(downcast_safe<Derived2>(std::move(ptr)));
            REQUIRE_UNARY(ptr);
        }
    }
}

TEST_SUITE_END();
