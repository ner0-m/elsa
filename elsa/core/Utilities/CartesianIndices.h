#pragma once

#include "TypeCasts.hpp"
#include "elsaDefines.h"
#include "Error.h"

#include <Eigen/Core>
#include <algorithm>
#include <vector>

#include "spdlog/fmt/fmt.h"

namespace elsa
{
    using IndexRange = std::pair<int, int>;

    /*
     * @brief This is a iterable view over a Cartesian index space. This defines a rectangular
     * region of integer indices. This is mostly equivalent to a n-dimensional loop, e.g.:
     *
     * ```cpp
     * for(int i = istart; i < istop; ++i) {
     *     for(int j = jstart; j < jstop; ++j) {
     *         // nested arbitrarily deep
     *     }
     * }
     * ```
     *
     * If you want to iterate over all indices form `0` to `upper` in a n-dimensional setting you
     * can:
     *
     * ```cpp
     * for(auto pos : CartesianIndex(upper)) {
     *     // nested arbitrarily deep
     * }
     * ```
     *
     * Note, that this is as typically an exclusive range for the end, i.e. [lower, upper).
     * At no point any coefficient of `pos` take the value of `upper` (lower[i] <= pos[i] < upper[i]
     * for all i in 0, ... pos.size()).
     *
     * Assuming `x` is a vector of integers. If you don't want to start at `0`. Pass the lower bound
     * index as the first argument:
     *
     * ```cpp
     * for(auto pos : CartesianIndex(lower, upper)) {
     *     // nested arbitrarily deep
     * }
     * ```
     *
     * The behaviour is quite similar to `std::views::iota` in C++20, but extended to n dimensions.
     * In this sense it is quite similar to Julies, check this
     * [blob](https://julialang.org/blog/2016/02/iteration/) for more information. This was also
     * the initial inspiration for this implementation.
     *
     */
    class CartesianIndices
    {
    private:
        index_t size_;
        IndexVector_t first_;
        IndexVector_t last_;

        // Tag struct, just to indicate the end iterator as sentinel. TODO: With a switch to C++
        // 20, remove this and let `end()` return a proper sentinel type
        struct as_sentinel {
        };

        static constexpr as_sentinel as_sentinel_t{};

    public:
        /// Vector of pairs, which represent the range for each dimension
        CartesianIndices(std::vector<IndexRange> ranges);

        /// Create an index space ranging from `0`, to `to` with the dimension of `to`
        CartesianIndices(const IndexVector_t& to);

        template <typename Vector>
        CartesianIndices(const Vector& to)
            : size_(to.size()), first_(IndexVector_t::Zero(size_)), last_(to.size())
        {
            using std::begin;
            std::copy_n(begin(to), size_, last_.begin());
        }

        /// Create an index space ranging from `from`, to `to` with the dimension of `ranges`
        CartesianIndices(const IndexVector_t& from, const IndexVector_t& to);

        template <typename Vector>
        CartesianIndices(const Vector& from, const Vector& to)
            : size_(to.size()), first_(from.size()), last_(to.size())
        {
            using std::begin;
            std::copy_n(begin(from), size_, first_.begin());
            std::copy_n(begin(to), size_, last_.begin());
        }

        /// Return the dimension of index space
        auto dims() const -> index_t;

        /// Return the number of coordinates of the space
        auto size() const -> index_t;

        /// return a range for a given dimension (0 <= i < dims())
        auto range(index_t i) const -> IndexRange;

        /// Return the lower bound of the index space
        auto first() -> IndexVector_t&;
        /// @overload
        auto first() const -> const IndexVector_t&;

        /// Return the upper bound of the index space
        auto last() -> IndexVector_t&;
        /// @overload
        auto last() const -> const IndexVector_t&;

        /// Random Access Iterator for index space
        struct iterator {
        private:
            IndexVector_t cur_;
            const IndexVector_t& begins_;
            const IndexVector_t& ends_;

        public:
            using value_type = IndexVector_t;
            using reference = value_type&;
            using pointer = value_type*;
            using difference_type = std::ptrdiff_t;
            using iterator_category = std::random_access_iterator_tag;

            iterator(as_sentinel, const IndexVector_t& begin, const IndexVector_t& end);

            explicit iterator(const IndexVector_t& vec);

            iterator(const IndexVector_t& cur, const IndexVector_t& begin,
                     const IndexVector_t& end);

            // Dereference
            const IndexVector_t& operator*() const;

            // Increment
            iterator& operator++();
            iterator operator++(int);

            // decrement
            iterator& operator--();
            iterator operator--(int);

            // advance
            auto operator+=(difference_type n) -> CartesianIndices::iterator&;
            auto operator-=(difference_type n) -> CartesianIndices::iterator&;

            friend auto operator+(const iterator& iter, difference_type n) -> iterator;
            friend auto operator+(difference_type n, const iterator& iter) -> iterator;

            friend auto operator-(const iterator& iter, difference_type n) -> iterator;
            friend auto operator-(const iterator& lhs, const iterator& rhs) -> difference_type;

            auto operator[](difference_type n) const -> value_type;

            // comparison
            friend auto operator==(const iterator& lhs, const iterator& rhs) -> bool
            {
                return lhs.cur_ == rhs.cur_;
            }

            friend auto operator!=(const iterator& lhs, const iterator& rhs) -> bool
            {
                return !(lhs == rhs);
            }

            // relational operators
            friend auto operator<(const iterator& lhs, const iterator& rhs) -> bool;
            friend auto operator>(const iterator& lhs, const iterator& rhs) -> bool;
            friend auto operator<=(const iterator& lhs, const iterator& rhs) -> bool;
            friend auto operator>=(const iterator& lhs, const iterator& rhs) -> bool;

        private:
            auto at_end() -> bool;
            auto at_end() const -> bool;

            // TODO: Prefer a for loop, and remove recursive function
            auto inc_recursive(index_t N) -> void;
            auto inc() -> void;

            auto prev_recursive(index_t N) -> void;
            auto prev() -> void;

            auto distance_to_recusrive(const iterator& iter, difference_type N) const
                -> difference_type;
            auto distance_to(const iterator& iter) const -> difference_type;

            auto advance_recursive(difference_type n, index_t N) -> void;
            auto advance(difference_type n) -> void;
        };

        /// Return the begin iterator to the index space
        auto begin() -> iterator;
        /// Return the end iterator to the index space
        auto end() -> iterator;

        /// @overload
        auto begin() const -> iterator;
        /// @overload
        auto end() const -> iterator;
    };

    /// @brief Visit all neighbours with a certain distance `dist`, which have the same x-axis (i.e.
    /// they lie in the same y-z plane)
    ///
    /// The returned position can lie outside of the volume or have negative indices, so be careful
    CartesianIndices neighbours_in_slice(const IndexVector_t& pos, const IndexVector_t& dist);

    /// @overload
    CartesianIndices neighbours_in_slice(const IndexVector_t& pos, index_t dist);

    /// @brief Visit all neighbours with a certain distance `dist`, which have the same x-axis (i.e.
    /// they lie in the same y-z plane), plus make sure we stay in bounds of `lower`, and `upper`.
    ///
    /// Given a certain voxel in a volume, you can visit all close by voxels, which are still in the
    /// volume
    ///
    /// ```cpp
    /// auto volmin = {0, 0, 0};
    /// auto volmax = {128, 128, 128};
    /// auto current_pos = ...;
    /// for(auto neighbours : neighbours_in_slice(current_pos, 1, volmin, volmax)) {
    ///     // ...
    /// }
    /// ```
    /// Note, that the current_pos is also visited!
    CartesianIndices neighbours_in_slice(const IndexVector_t& pos, const IndexVector_t& dist,
                                         const IndexVector_t& lower, const IndexVector_t& upper);

    /// @overload
    CartesianIndices neighbours_in_slice(const IndexVector_t& pos, index_t dist, index_t leadingDim,
                                         const IndexVector_t& lower, const IndexVector_t& upper);
} // namespace elsa

template <>
struct fmt::formatter<elsa::CartesianIndices> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const elsa::CartesianIndices& idx, FormatContext& ctx)
    {
        fmt::format_to(ctx.out(), "(");

        for (int i = 0; i < idx.dims() - 1; ++i) {
            auto p = idx.range(i);
            fmt::format_to(ctx.out(), "{}:{}, ", p.first, p.second);
        }
        auto p = idx.range(idx.dims() - 1);
        return fmt::format_to(ctx.out(), "{}:{})", p.first, p.second);
    }
};
