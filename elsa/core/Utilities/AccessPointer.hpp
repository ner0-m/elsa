#pragma once

#include <memory>
#include <cassert>

namespace elsa
{
    namespace detail
    {
        template <typename T>
        bool unique(const std::shared_ptr<T>& ptr)
        {
            return ptr.use_count() == 1;
        }
    } // namespace detail

    /**
     * @brief Smart pointer abstraction, which models different possible access modes. Overall, the
     * design is quite close to standard pointer behaviour, but write access can only be requested
     * via the `write()` method. All other access pointer (dereference, ->, and `read()`) only
     * return a const reference to the element type.
     *
     * The behaviour for `read()` and `write()` are depending on the policy template. The two use
     * cases for us are copy-on-write and observing behaviour.
     *
     * - copy-on-write: As the name suggests, if you request write access to the pointer a copy of
     * the pointed to object will be created if the reference is not shared by any other.
     * - observer: Very similar to a raw pointer, read and write will not create a copy, and writing
     * to a shared object, will just change the object and not trigger a copy (compared to
     * copy-on-write)
     *
     * Other options would be possible, but are not of use to use. The class calls the access
     * policies provided `read()` and `write()` functions before any access to the pointed to
     * object.
     *
     * Note, be aware that the element type is expected to be Regular, as in
     * [std::regular](https://en.cppreference.com/w/cpp/concepts/regular)
     *
     * References:
     *   - https://stlab.cc/legacy/copy-on-write.html
     *   - https://stlab.cc/libraries/stlab2Fcopy_on_write.hpp/copy_on_write3CT3E/
     *   - https://stlab.adobe.com/group__concept__regular__type.html
     *
     * @author David Frank - Initial design
     *
     * @tparam T - element type of pointed to object (models regular)
     * @tparam AccessPolicy - decide behaviour for read and write access
     *
     */
    template <typename T, typename AccessPolicy>
    class AccessPointer
    {
        /// Keep a strong reference to managed object
        std::shared_ptr<T> self_;

        /// accessor, managing access behaviour
        /// TODO: Use empty base class optimization or C++20's [[no_unique_address]]
        AccessPolicy accessor_;

        /// helper to disable copy constructor for single argument value constructor
        template <class U>
        using disable_copy =
            std::enable_if_t<!std::is_same<std::decay_t<U>, AccessPointer>::value>*;

        /// helper to disable copy constructor for single argument value assignment operator
        template <typename U>
        using disable_copy_assign =
            std::enable_if_t<!std::is_same<std::decay_t<U>, AccessPointer>::value, AccessPointer&>;

    public:
        using element_type = T;

        using access_policy = AccessPolicy;

        /// Default constructor
        AccessPointer() = default;

        template <class U>
        AccessPointer(U&& x, disable_copy<U> = nullptr)
            : self_(std::make_shared<T>(std::forward<U>(x)))
        {
        }

        template <class U, class V, class... Args>
        AccessPointer(U&& x, V&& y, Args&&... args)
            : self_(std::make_shared<T>(std::forward<U>(x), std::forward<V>(y),
                                        std::forward<Args>(args)...))
        {
        }

        /// Assign from value
        template <class U>
        auto operator=(U&& x) -> disable_copy_assign<U>
        {
            assert(self_ && "FATAL: using a moved AccessPointer object");

            accessor_.write(self_);
            *self_ = std::forward<U>(x);

            return *this;
        }

        auto write() -> element_type&
        {
            assert(self_ && "FATAL: using a moved AccessPointer object");

            accessor_.write(self_);
            return *self_;
        }

        auto read() const noexcept -> const element_type&
        {
            assert(self_ && "FATAL: using a moved AccessPointer object");

            accessor_.read(self_);

            return *self_;
        }

        auto operator*() const noexcept -> const element_type& { return read(); }

        auto operator->() const noexcept -> const element_type* { return &read(); }

        explicit operator bool() const noexcept { return static_cast<bool>(self_); }

        auto use_count() const noexcept
        {
            assert(self_ && "FATAL: using a moved AccessPointer object");

            return self_.use_count();
        }

        bool unique() const noexcept
        {
            assert(self_ && "FATAL: using a moved AccessPointer object");

            return detail::unique(self_);
        }

        /// Check if the pointers are equal, i.e. they point to the same object
        bool identity(const AccessPointer& x) const noexcept
        {
            assert((self_ && x.self_) && "FATAL: using a moved AccessPointer object");

            return self_ == x.self_;
        }

        friend void swap(AccessPointer& x, AccessPointer& y) noexcept
        {
            std::swap(x.self_, y.self_);
        }

        /// Less than operator, deep comparison (compare elements, not pointers)
        friend bool operator<(const AccessPointer& x, const AccessPointer& y) noexcept
        {
            return !x.identity(y) && (*x < *y);
        }

        friend bool operator<(const AccessPointer& x, const element_type& y) noexcept
        {
            return *x < y;
        }

        friend bool operator<(const element_type& x, const AccessPointer& y) noexcept
        {
            return x < *y;
        }

        friend bool operator>(const AccessPointer& x, const AccessPointer& y) noexcept
        {
            return y < x;
        }

        friend bool operator>(const AccessPointer& x, const element_type& y) noexcept
        {
            return y < x;
        }

        friend bool operator>(const element_type& x, const AccessPointer& y) noexcept
        {
            return y < x;
        }

        friend bool operator<=(const AccessPointer& x, const AccessPointer& y) noexcept
        {
            return !(y < x);
        }

        friend bool operator<=(const AccessPointer& x, const element_type& y) noexcept
        {
            return !(y < x);
        }

        friend bool operator<=(const element_type& x, const AccessPointer& y) noexcept
        {
            return !(y < x);
        }

        friend bool operator>=(const AccessPointer& x, const AccessPointer& y) noexcept
        {
            return !(x < y);
        }

        friend bool operator>=(const AccessPointer& x, const element_type& y) noexcept
        {
            return !(x < y);
        }

        friend bool operator>=(const element_type& x, const AccessPointer& y) noexcept
        {
            return !(x < y);
        }

        friend bool operator==(const AccessPointer& x, const AccessPointer& y) noexcept
        {
            return x.identity(y) || (*x == *y);
        }

        friend bool operator==(const AccessPointer& x, const element_type& y) noexcept
        {
            return *x == y;
        }

        friend bool operator==(const element_type& x, const AccessPointer& y) noexcept
        {
            return x == *y;
        }

        friend bool operator!=(const AccessPointer& x, const AccessPointer& y) noexcept
        {
            return !(x == y);
        }

        friend bool operator!=(const AccessPointer& x, const element_type& y) noexcept
        {
            return !(x == y);
        }

        friend bool operator!=(const element_type& x, const AccessPointer& y) noexcept
        {
            return !(x == y);
        }
    };

    /**
     * @brief Copy on write policy, will create a copy for write accesses, and do nothing for read
     * access requests
     *
     * NOTE: The read and write methods take a shared_ptr by reference, as they possibly re-seat the
     * pointer.
     */
    template <typename T>
    struct CopyOnWritePolicy {
        void read(const std::shared_ptr<T>&) const {}

        void write(std::shared_ptr<T>& ptr) const
        {
            if (!detail::unique(ptr)) {
                ptr = std::make_shared<T>(*ptr);
            }
        }
    };

    /**
     * @brief Observer policy, does nothing for neither read nor right access
     *
     * NOTE: The read and write methods take a shared_ptr by reference, as they possibly re-seat the
     * pointer.
     */
    template <typename T>
    struct ObserverPolicy {
        using element_type = T;

        void read(const std::shared_ptr<T>&) const {}

        void write(std::shared_ptr<T>&) const {}
    };

    template <typename T>
    using CopyOnWritePointer = AccessPointer<T, CopyOnWritePolicy<T>>;

    template <typename T>
    using ObserverPointer = AccessPointer<T, ObserverPolicy<T>>;
} // namespace elsa
