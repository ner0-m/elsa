#pragma once

#include <iterator>

namespace elsa::detail
{
    /**
     * \brief iterator which uses a non-owning raw pointer to iterate over a container. The iterator
     * is random access and assumes contiguous memory layout.
     *
     * \author David Frank - initial implementation
     *
     * \tparam T - the type of the container
     *
     * Note: comparing iterators from different containers is undefined behavior, so we do not check
     * for it.
     */
    template <typename T>
    class PtrIterator
    {
    public:
        /// alias for iterator type
        using self_type = PtrIterator;

        /// the iterator category
        using iterator_category = std::random_access_iterator_tag;
        /// the value type of container elements
        using value_type = typename T::value_type;
        /// pointer type of container elements
        using pointer = typename T::pointer;
        /// reference type of container elements
        using reference = typename T::reference;
        /// difference type of container
        using difference_type = typename T::difference_type;

        /// constructor taking a non-owning pointer to the data
        explicit PtrIterator(pointer ptr) : _ptr(ptr) {}

        /// de-referencing operator
        reference operator*() { return *_ptr; }
        /// pointer access operator
        pointer operator->() { return _ptr; }
        /// subscript operator
        reference operator[](int m) { return _ptr[m]; }

        /// prefix increment operator
        self_type operator++()
        {
            ++_ptr;
            return *this;
        }
        /// postfix increment operator
        self_type operator++(int) &
        {
            auto i = *this;
            ++_ptr;
            return i;
        }

        /// prefix decrement operator
        self_type operator--()
        {
            --_ptr;
            return *this;
        }
        /// postfix decrement operator
        self_type operator--(int) &
        {
            auto i = *this;
            --_ptr;
            return i;
        }

        /// moving iterator forward by n
        self_type& operator+=(int n)
        {
            _ptr += n;
            return *this;
        }
        /// moving iterator backward by n
        self_type& operator-=(int n)
        {
            _ptr -= n;
            return *this;
        }

        /// return new iterator moved forward by n
        self_type operator+(int n) const
        {
            self_type r(*this);
            return r += n;
        }
        /// return new iterator moved backward by n
        self_type operator-(int n) const
        {
            self_type r(*this);
            return r -= n;
        }

        /// return the difference between iterators
        difference_type operator-(PtrIterator const& r) const { return _ptr - r._ptr; }

        /// compare < with other iterator
        bool operator<(const PtrIterator& r) const { return _ptr < r._ptr; }
        /// compare <= with other iterator
        bool operator<=(const PtrIterator& r) const { return _ptr <= r._ptr; }
        /// compare > with other iterator
        bool operator>(const PtrIterator& r) const { return _ptr > r._ptr; }
        /// compare >= with other iterator
        bool operator>=(const PtrIterator& r) const { return _ptr >= r._ptr; }
        /// compare != with other iterator
        bool operator!=(const PtrIterator& r) const { return _ptr != r._ptr; }
        /// compare == with other iterator
        bool operator==(const PtrIterator& r) const { return _ptr == r._ptr; }

    private:
        /// non-owning (!) pointer to data (do not clean up or anything)
        pointer _ptr{};
    };

    /**
     * \brief constant iterator which uses a non-owning raw pointer to iterate over a container. The
     * iterator is random access and assumes contiguous memory layout. It is const in the sense that
     * it cannot mutate the state of the object it iterates over.
     *
     * \author David Frank - initial implementation
     *
     * \tparam T - the type of the container
     *
     * Note: comparing iterators from different containers is undefined behavior, so we do not check
     * for it.
     */
    template <typename T>
    class ConstPtrIterator
    {
    public:
        /// alias for iterator type
        using self_type = ConstPtrIterator;

        /// the iterator category
        using iterator_category = std::random_access_iterator_tag;
        /// the value type of container elements
        using value_type = typename T::value_type;
        /// pointer type of container elements
        using pointer = typename T::const_pointer;
        /// reference type of container elements
        using reference = typename T::const_reference;
        /// difference type of container
        using difference_type = typename T::difference_type;

        /// constructor taking a non-owning pointer to the data
        explicit ConstPtrIterator(pointer ptr) : _ptr(ptr) {}

        /// de-referencing operator
        reference operator*() { return *_ptr; }
        /// pointer access operator
        pointer operator->() { return _ptr; }
        /// subscript operator
        reference operator[](int m) { return _ptr[m]; }

        /// prefix increment operator
        self_type operator++()
        {
            ++_ptr;
            return *this;
        }
        /// postfix increment operator
        self_type operator++(int) &
        {
            auto i = *this;
            ++_ptr;
            return i;
        }

        /// prefix decrement operator
        self_type operator--()
        {
            --_ptr;
            return *this;
        }
        /// postfix decrement operator
        self_type operator--(int) &
        {
            auto i = *this;
            --_ptr;
            return i;
        }

        /// moving iterator forward by n
        self_type& operator+=(int n)
        {
            _ptr += n;
            return *this;
        }
        /// moving iterator backward by n
        self_type& operator-=(int n)
        {
            _ptr -= n;
            return *this;
        }

        /// return new iterator moved forward by n
        self_type operator+(int n) const
        {
            self_type r(*this);
            return r += n;
        }
        /// return new iterator moved backward by n
        self_type operator-(int n) const
        {
            self_type r(*this);
            return r -= n;
        }

        /// return the difference between iterators
        difference_type operator-(ConstPtrIterator const& r) const { return _ptr - r._ptr; }

        /// compare < with other iterator
        bool operator<(const self_type& r) const { return _ptr < r._ptr; }
        /// compare <= with other iterator
        bool operator<=(const self_type& r) const { return _ptr <= r._ptr; }
        /// compare > with other iterator
        bool operator>(const self_type& r) const { return _ptr > r._ptr; }
        /// compare >= with other iterator
        bool operator>=(const self_type& r) const { return _ptr >= r._ptr; }
        /// compare != with other iterator
        bool operator!=(const self_type& r) const { return _ptr != r._ptr; }
        /// compare == with other iterator
        bool operator==(const self_type& r) const { return _ptr == r._ptr; }

    private:
        /// non-owning (!) pointer to data (do not clean up or anything)
        pointer _ptr{};
    };

} // end namespace elsa::detail

namespace elsa
{
    /// alias for the iterator for DataContainer
    template <typename T>
    using DataContainerIterator = detail::PtrIterator<T>;

    /// alias for the constant iterator for DataContainer
    template <typename T>
    using ConstDataContainerIterator = detail::ConstPtrIterator<T>;
} // end namespace elsa
