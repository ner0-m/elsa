#include "CartesianIndices.h"
#include "TypeCasts.hpp"
#include <iostream>
#include <iterator>
#include <numeric>

namespace elsa
{
    template <typename Fn>
    auto map(std::vector<IndexRange> v, Fn fn)
    {
        IndexVector_t result(v.size());
        std::transform(v.begin(), v.end(), result.begin(), std::move(fn));
        return result;
    }

    auto CartesianIndices::dims() const -> index_t { return as<index_t>(idxrange_.size()); }

    auto CartesianIndices::size() const -> index_t
    {
        // TODO: Once the coverage CI image is updated, change this to:
        // clang-format off
        // std::transform_reduce(idxrange_.begin(), idxrange_.end(), 1, std::multiplies<>{},
        //                      [](auto p) { return p.second - p.first; }));
        // clang-format on
        std::vector<index_t> tmp;
        tmp.reserve(idxrange_.size());

        std::transform(idxrange_.begin(), idxrange_.end(), std::back_inserter(tmp),
                       [](auto p) { return p.second - p.first; });

        return std::accumulate(tmp.begin(), tmp.end(), 1, std::multiplies<>{});
    }

    auto CartesianIndices::range(index_t i) const -> IndexRange { return idxrange_[asUnsigned(i)]; }

    auto CartesianIndices::first() -> IndexVector_t
    {
        return map(idxrange_, [](auto range) { return range.first; });
    }

    auto CartesianIndices::first() const -> IndexVector_t
    {
        return map(idxrange_, [](auto range) { return range.first; });
    }

    auto CartesianIndices::last() -> IndexVector_t
    {
        return map(idxrange_, [](auto range) { return range.second; });
    }

    auto CartesianIndices::last() const -> IndexVector_t
    {
        return map(idxrange_, [](auto range) { return range.second; });
    }

    auto CartesianIndices::begin() -> iterator { return iterator(first(), first(), last()); }

    auto CartesianIndices::begin() const -> iterator { return {first(), first(), last()}; }

    auto CartesianIndices::end() -> iterator { return {as_sentinel_t, first(), last()}; }

    auto CartesianIndices::end() const -> iterator { return {as_sentinel_t, first(), last()}; }

    /****** Iterator implementation ******/
    CartesianIndices::iterator::iterator(as_sentinel, const IndexVector_t& begin,
                                         const IndexVector_t& end)
        : cur_(end), begins_(end), ends_(end)
    {
        cur_.tail(cur_.size() - 1) = begin.tail(cur_.size() - 1);
    }

    CartesianIndices::iterator::iterator(IndexVector_t vec) : cur_(vec), begins_(vec), ends_(vec) {}

    CartesianIndices::iterator::iterator(IndexVector_t cur, IndexVector_t begin, IndexVector_t end)
        : cur_(cur), begins_(begin), ends_(end)
    {
    }

    auto CartesianIndices::iterator::operator*() const -> IndexVector_t { return cur_; }

    auto CartesianIndices::iterator::operator++() -> iterator&
    {
        inc();
        return *this;
    }

    auto CartesianIndices::iterator::operator++(int) -> iterator
    {
        auto copy = *this;
        ++(*this);
        return copy;
    }

    auto CartesianIndices::iterator::operator--() -> iterator&
    {
        prev();
        return *this;
    }

    auto CartesianIndices::iterator::operator--(int) -> iterator
    {
        auto copy = *this;
        --(*this);
        return copy;
    }

    auto CartesianIndices::iterator::operator+=(difference_type n) -> iterator&
    {
        advance(n);
        return *this;
    }

    auto CartesianIndices::iterator::operator-=(difference_type n) -> iterator&
    {
        advance(-n);
        return *this;
    }

    auto operator+(const CartesianIndices::iterator& iter,
                   CartesianIndices::iterator::difference_type n) -> CartesianIndices::iterator
    {
        auto copy = iter;
        copy += n;
        return copy;
    }

    auto operator+(CartesianIndices::iterator::difference_type n,
                   const CartesianIndices::iterator& iter) -> CartesianIndices::iterator
    {
        return iter + n;
    }

    auto operator-(const CartesianIndices::iterator& iter,
                   CartesianIndices::iterator::difference_type n) -> CartesianIndices::iterator
    {
        auto copy = iter;
        copy -= n;
        return copy;
    }

    auto operator-(const CartesianIndices::iterator& lhs, const CartesianIndices::iterator& rhs)
        -> CartesianIndices::iterator::difference_type
    {
        return rhs.distance_to(lhs);
    }

    auto CartesianIndices::iterator::operator[](difference_type n) const -> value_type
    {
        // TODO: Make this more efficient
        auto copy = *this;
        copy += n;
        return *copy;
    }

    auto operator<(const CartesianIndices::iterator& lhs, const CartesianIndices::iterator& rhs)
        -> bool
    {
        // Not sure why I have to instantiate them this way, else it doesn't work...
        using Array = Eigen::Array<index_t, Eigen::Dynamic, 1>;
        const Array a1 = lhs.cur_.array();
        const Array a2 = rhs.cur_.array();

        return (a1 < a2).any();
    }

    auto operator>(const CartesianIndices::iterator& lhs, const CartesianIndices::iterator& rhs)
        -> bool
    {
        return rhs < lhs;
    }

    auto operator<=(const CartesianIndices::iterator& lhs, const CartesianIndices::iterator& rhs)
        -> bool
    {
        return !(lhs > rhs);
    }

    auto operator>=(const CartesianIndices::iterator& lhs, const CartesianIndices::iterator& rhs)
        -> bool
    {
        return !(lhs < rhs);
    }

    auto CartesianIndices::iterator::at_end() -> bool { return cur_[0] == ends_[0]; }

    auto CartesianIndices::iterator::at_end() const -> bool { return cur_[0] == ends_[0]; }

    auto CartesianIndices::iterator::inc_recursive(index_t N) -> void
    {
        auto& iter = cur_[N];

        if (++iter == ends_[N] && N > 0) {
            iter = begins_[N];
            inc_recursive(N - 1);
        }
    }

    auto CartesianIndices::iterator::inc() -> void { inc_recursive(cur_.size() - 1); }

    auto CartesianIndices::iterator::prev_recursive(index_t N) -> void
    {
        auto& iter = cur_[N];

        if (iter == begins_[N] && N > 0) {
            iter = ends_[N];
            inc_recursive(N - 1);
        }
        --iter;
    }

    auto CartesianIndices::iterator::prev() -> void { prev_recursive(cur_.size() - 1); }

    auto CartesianIndices::iterator::distance_to_recusrive(const iterator& other,
                                                           difference_type N) const
        -> difference_type
    {
        if (N == 0) {
            return other.cur_[N] - cur_[N];
        } else {
            const auto d = distance_to_recusrive(other, N - 1);
            const auto scale = ends_[N] - begins_[N];
            const auto increment = other.cur_[N] - cur_[N];

            return difference_type{(d * scale) + increment};
        }
    }

    auto CartesianIndices::iterator::distance_to(const iterator& other) const -> difference_type
    {
        auto N = cur_.size() - 1;
        return distance_to_recusrive(other, N);
    }

    auto CartesianIndices::iterator::advance_recursive(difference_type n, index_t N) -> void
    {
        if (n == 0) {
            return;
        }

        auto& iter = cur_[N];
        const auto size = ends_[N] - begins_[N];
        const auto first = begins_[N];

        auto const idx = as<difference_type>(iter - first);
        n += idx;

        auto div = size ? n / size : 0;
        auto mod = size ? n % size : 0;

        if (N != 0) {
            if (mod < 0) {
                mod += size;
                div--;
            }
            advance_recursive(div, N - 1);
        } else {
            if (div > 0) {
                mod = size;
            }
        }

        iter = first + mod;
    }

    auto CartesianIndices::iterator::advance(difference_type n) -> void
    {
        advance_recursive(n, cur_.size() - 1);
    }

    CartesianIndices neighbours_in_slice(const IndexVector_t& pos, const IndexVector_t& dist)
    {
        Eigen::IOFormat format(4, 0, ", ", "", "", "", "[", "]");
        IndexVector_t from = pos;
        from.tail(pos.size() - 1).array() -= dist.array();

        IndexVector_t to = pos;
        to.tail(pos.size() - 1).array() += dist.array() + 1;
        to[0] += 1; // Be sure this is incremented, so we actually iterate over it

        return {from, to};
    }

    CartesianIndices neighbours_in_slice(const IndexVector_t& pos, index_t dist)
    {
        return neighbours_in_slice(pos, IndexVector_t::Constant(pos.size() - 1, dist));
    }

    CartesianIndices neighbours_in_slice(const IndexVector_t& pos, const IndexVector_t& dist,
                                         const IndexVector_t& lower, const IndexVector_t& upper)
    {
        IndexVector_t from = pos;
        from.array() -= dist.array();
        from = from.cwiseMax(lower);

        IndexVector_t to = pos;
        to.array() += dist.array() + 1;
        to = to.cwiseMin(upper);

        // FIXME: sometimes this happens when padding is added to an aabb
        from = (to.array() == from.array()).select(from.array() - 1, from);

        return {from, to};
    }

    CartesianIndices neighbours_in_slice(const IndexVector_t& pos, index_t dist, index_t leadingDim,
                                         const IndexVector_t& lower, const IndexVector_t& upper)
    {
        IndexVector_t distvec = IndexVector_t::Constant(pos.size(), dist);
        distvec[leadingDim] = 0;

        return neighbours_in_slice(pos, distvec, lower, upper);
    }
} // namespace elsa
