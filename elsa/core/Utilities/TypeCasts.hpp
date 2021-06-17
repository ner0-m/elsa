#pragma once

#include <type_traits>
#include <limits>
#include <cassert>
#include <memory>

#include "elsaDefines.h"

namespace elsa
{
    /// Check if a type can be dynamically casted to another
    template <typename Derived, typename Base>
    bool is(Base& input)
    {
        using OutputType =
            std::conditional_t<std::is_const_v<Base>, std::add_const_t<Derived>, Derived>;
        return dynamic_cast<OutputType*>(&input);
    }

    /// Overload to check for pointer types directly
    template <typename Derived, typename Base>
    bool is(Base* input)
    {
        return input && is<Derived>(*input);
    }

    /// Overload to check for unique_ptr directly.
    /// Usually passing a const reference to a unique_ptr is really dumb, but for this case it's
    /// what is needed.
    template <typename Derived, typename Base, typename Deleter>
    bool is(std::unique_ptr<Base, Deleter>& input)
    {
        return input && is<Derived>(*input);
    }

    /// Downcast pointer, assumes that type is known (i.e. checked by is(...))
    template <typename Derived, typename Base>
    auto downcast(Base* input)
        -> std::conditional_t<std::is_const_v<Base>, std::add_const_t<Derived>, Derived>*
    {
        static_assert(std::is_base_of_v<Base, Derived>, "To downcast, types needs to be derived");
        using ReturnType =
            std::conditional_t<std::is_const_v<Base>, std::add_const_t<Derived>, Derived>;

        assert(!input || is<Derived>(*input));

        return static_cast<ReturnType*>(input);
    }

    /// Downcast reference, assumes that type is known (i.e. checked by is(...))
    template <typename Derived, typename Base>
    auto downcast(Base& input)
        -> std::conditional_t<std::is_const_v<Base>, std::add_const_t<Derived>, Derived>&
    {
        static_assert(std::is_base_of_v<Base, Derived>, "To downcast, types needs to be derived");
        using ReturnType =
            std::conditional_t<std::is_const_v<Base>, std::add_const_t<Derived>, Derived>;

        assert(is<Derived>(input));

        return static_cast<ReturnType&>(input);
    }

    /// Downcast reference, assumes that type is known (i.e. checked by is(...))
    template <typename Derived, typename Base, typename Deleter>
    std::unique_ptr<Derived, Deleter> downcast(std::unique_ptr<Base, Deleter>&& input)
    {
        static_assert(std::is_base_of_v<Base, Derived>, "To downcast, types needs to be derived");

        assert(is<Derived>(input));

        auto d = static_cast<Derived*>(input.release());
        return std::unique_ptr<Derived, Deleter>(d, std::move(input.get_deleter()));
    }

    /// Try to downcast pointer to Base to Derived, return a nullptr if it fails
    template <typename Derived, typename Base>
    Derived* downcast_safe(Base* input)
    {
        static_assert(std::is_base_of_v<Base, Derived>, "To downcast, types needs to be derived");

        if (is<Derived>(input)) {
            return downcast<Derived>(input);
        }

        return nullptr;
    }

    /// Try to downcast reference to Base to Derived, Will throw std::bad_cast if it can't
    /// dynamically cast to Derived
    template <typename Derived, typename Base>
    Derived& downcast_safe(Base& input)
    {
        static_assert(std::is_base_of_v<Base, Derived>, "To downcast, types needs to be derived");

        return dynamic_cast<Derived&>(input);
    }

    /// Try to downcast a unique_ptr to Base to Derived, return a nullptr if it fails
    /// Note that if the downcast can't be performed the callees unique_ptr is not touched
    /// But once the cast is done, the callees unique_ptr is not safe to use anymore
    template <typename Derived, typename Base, typename Deleter>
    std::unique_ptr<Derived, Deleter> downcast_safe(std::unique_ptr<Base, Deleter>&& input)
    {
        static_assert(std::is_base_of_v<Base, Derived>, "To downcast, types needs to be derived");

        if (Derived* result = downcast_safe<Derived>(input.get())) {
            input.release();
            return std::unique_ptr<Derived, Deleter>(result, std::move(input.get_deleter()));
        }

        return std::unique_ptr<Derived, Deleter>(nullptr, input.get_deleter());
    }

    namespace detail
    {
        /// Alias to remove const, volatile and reference qualifiers of type
        template <typename T>
        using remove_crv_t = std::remove_reference_t<std::remove_cv_t<T>>;
    } // namespace detail

    /// Cast from one type to another without any checks, use with care
    template <typename To, typename From>
    auto as(From&& from) noexcept
    {
        return static_cast<To>(std::forward<From>(from));
    }

    /// Convert a signed value to an unsigned. Check for underflow only
    template <typename From>
    auto asUnsigned(From&& v) noexcept
    {
        static_assert(std::is_arithmetic_v<detail::remove_crv_t<From>>,
                      "Expect arithmetic type (use as() instead)");

        if constexpr (std::is_unsigned_v<From>) {
            return std::forward<From>(v);
        }

        using To = std::make_unsigned_t<detail::remove_crv_t<From>>;

        assert(v >= 0 && "Only convert positive numbers to an unsigned");
        return as<To>(std::forward<From>(v));
    }

    /// Convert an unsigned value to an signed. Check for overflow only
    template <typename From>
    auto asSigned(From&& v) noexcept
    {
        static_assert(std::is_arithmetic_v<detail::remove_crv_t<From>>,
                      "Expect arithmetic type (use as() instead)");

        if constexpr (std::is_signed_v<From>) {
            return std::forward<From>(v);
        }

        using To = std::make_signed_t<detail::remove_crv_t<From>>;

        assert(v <= std::numeric_limits<To>::max() && "Converted value is overflown");
        return as<To>(std::forward<From>(v));
    }
} // namespace elsa
