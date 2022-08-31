#pragma once

#include <optional>

namespace elsa
{
    /// @brief A very tiny wrapper around std::optional to give it a little nicer name for this
    /// specific use case
    template <class T>
    class MaybeUninitialized
    {
    public:
        MaybeUninitialized() = default;

        /// Uninitialized can be constructed using T
        explicit MaybeUninitialized(const T& x) : value_(x) {}
        explicit MaybeUninitialized(T&& x) : value_(std::move(x)) {}

        /// Uninitialized can be assigned using T
        MaybeUninitialized& operator=(const T& x)
        {
            value_ = x;
            return *this;
        }

        MaybeUninitialized& operator=(T&& x)
        {
            value_ = std::move(x);
            return *this;
        }

        bool isInitialized() const { return value_.has_value(); }

        T& operator*() { return *value_; }
        const T& operator*() const { return *value_; }

    private:
        std::optional<T> value_ = std::nullopt;
    };

    template <class T>
    bool operator==(const MaybeUninitialized<T>& lhs, const MaybeUninitialized<T>& rhs)
    {
        if (!lhs.isInitialized() && !rhs.isInitialized()) {
            return true;
        } else if (lhs.isInitialized() && rhs.isInitialized()) {
            return *lhs == *rhs;
        } else {
            return false;
        }
    }

    template <class T>
    bool operator!=(const MaybeUninitialized<T>& lhs, const MaybeUninitialized<T>& rhs)
    {
        return !(lhs == rhs);
    }
} // namespace elsa
