#include "SetupHelpers.h"

namespace elsa
{
    namespace detail
    {
        RangeIterator::RangeIterator(int cur, int step, bool isEnd)
            : _cur(cur), _step(step), _isEnd(isEnd)
        {
        }

        RangeIterator RangeIterator::operator++()
        {
            advance();
            return *this;
        }

        RangeIterator RangeIterator::operator++(int)
        {
            auto tmp = *this;
            advance();
            return tmp;
        }

        int RangeIterator::operator*() { return _cur; }

        bool RangeIterator::operator==(RangeIterator rhs) const { return !(*this != rhs); }
        bool RangeIterator::operator!=(RangeIterator rhs) const
        {
            // If both iterators are end, comparison is false
            if (_isEnd && rhs._isEnd)
                return false;

            // If none is an end, compare values
            if (!_isEnd && !rhs._isEnd)
                return _cur != rhs._cur;

            // We know that one is an end, check that the one not the end
            // is in the range of the other one
            return notEqualToEnd(*this, rhs);
        }

        template <typename T>
        constexpr bool RangeIterator::isWithinRange(T val, T stopVal, [[maybe_unused]] T stepVal)
        {
            if constexpr (std::is_unsigned<T>{}) {
                return val < stopVal;
            } else {
                return !(stepVal > 0 && val >= stopVal) && !(stepVal < 0 && val <= stopVal);
            }
        }

        bool RangeIterator::notEqualToEnd(const RangeIterator& lhs,
                                          const RangeIterator& rhs) noexcept
        {
            if (rhs._isEnd) {
                return isWithinRange(lhs._cur, rhs._cur, lhs._step);
            }
            return isWithinRange(rhs._cur, rhs._cur, rhs._step);
        }

        void RangeIterator::advance() { _cur += _step; }
    } // namespace detail

    detail::RangeIterator Range::begin() const
    {
        return detail::RangeIterator(_start, _step, false);
    }

    detail::RangeIterator Range::end() const { return detail::RangeIterator(_end, _step, true); }
} // namespace elsa
