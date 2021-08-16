#pragma once

namespace elsa
{
    /**
     * @brief Make public interfaces only callable by certain types.
     *
     *
     * If type T need insight into certain function of another class U,
     * but if the it doesn't make sense to make it publicly available to
     * all classes, you could make T a friend of U. But then access to the complete
     * private implementation is granted.
     *
     * This Badge types is some middle way. The specific member functions
     * needed by T are still public, but class T has to show its badge to get access
     * to it.
     *
     * This is specifically useful, if the private function is a private
     * constructor of U and used to call `std::make_unique` (such that only T
     * can create objects of class U with this specific constructor). Making
     * T a friend of U, doesn't help, as `std::make_unique` still can't access
     * the private implementation, but with the badge it works.
     *
     * This is a small example of the Badge:
     *
     * @code{.cpp}
     * class Bar; // Forward declaration
     *
     * class Foo {
     * public:
     *     void foo(Bar bar) {
     *         // This is a legal call
     *         bar.internal_needed_by_foo({});
     *     }
     * };
     *
     * class Bar {
     * public:
     *     void internal_needed_by_foo(Badge<Foo>)
     * };
     * @endcode
     *
     * Then `internal_needed_by_foo` is only callable from `Foo`, but no other class.
     *
     * References:
     * https://awesomekling.github.io/Serenity-C++-patterns-The-Badge/
     *
     * @author David Frank - initial code
     *
     * @tparam T
     */
    template <typename T>
    class Badge
    {
        /// Make T a friend of `Badge`
        friend T;

        /// Private constructor accessible by T
        Badge() = default;
    };
} // namespace elsa
