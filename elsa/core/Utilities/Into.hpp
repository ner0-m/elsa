#pragma once

#include <type_traits>
#include <utility>
#include <optional>

namespace elsa
{
    /**
     * Model after the rust into trait (https://doc.rust-lang.org/std/convert/trait.Into.html)
     * It is often used at trait boundaries to provide flexible API design (instead of overloading
     * for all kind of types).
     *
     * In this implementation, if a type U provides a conversion operator to the type T (`operator
     * T()`), then this can be used as a function argument, instead of overloading.
     *
     * Example:
     *
     * ```cpp
     * auto foo(Into<int> mytype) {
     *     auto i = mytype.into();
     * }
     * ```
     */
    template <typename T>
    class Into
    {
    public:
        template <typename U, typename = std::enable_if_t<std::is_convertible_v<U, T>>>
        Into(U&& into) : inner_(static_cast<T>(into))
        {
        }

        auto into() const -> T { return inner_; }

    private:
        T inner_;
    };

    template <typename T>
    class TryInto
    {
    public:
        template <typename U, std::enable_if_t<std::is_convertible_v<U, T>, int> = 0>
        TryInto(U&& into) : inner_(static_cast<T>(into))
        {
        }

        template <typename U, std::enable_if_t<!std::is_convertible_v<U, T>, int> = 0>
        TryInto(U&& /* into */) : inner_()
        {
        }

        auto tryInto() const -> std::optional<T> { return inner_; }

    private:
        std::optional<T> inner_{};
    };
} // namespace elsa
