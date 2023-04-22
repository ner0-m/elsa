#pragma once

#include "DisableWarnings.h"

DISABLE_WARNING_PUSH
DISABLE_WARNING_SIGN_CONVERSION
// Thrust is smart enough to always pick the correct vector for us
#include <thrust/universal_vector.h>
DISABLE_WARNING_POP

#include "memory_resource/ContiguousMemory.h"

#include <iterator>

/* if iterator is pointer, it must be continuous, hence optimizations can be made */

namespace elsa
{
    /*
     *   Used Both as Pointer Wrapper and Iterator (hence the iterator_category)
     *       -> Necessary for integration with thrust helper types
     */
    template <class Type>
    class ContiguousPointer
    {
    public:
        using self_type = ContiguousPointer<Type>;
        using pointer = Type*;
        using const_pointer = const Type*;
        using value_type = std::remove_cv_t<Type>;
        using reference = Type&;
        using const_reference = const Type&;
        using size_type = size_t;
        using difference_type = ptrdiff_t;
        using iterator_category = std::random_access_iterator_tag;

    private:
        pointer _where = 0;

    public:
        ContiguousPointer() {}
        ContiguousPointer(pointer w) : _where(w) {}
        ContiguousPointer(const self_type& p) : _where(p._where) {}
        ContiguousPointer(self_type&& p) noexcept : _where(p._where) {}

    public:
        /*
         *   for compatability with thrust
         */
        using raw_pointer = pointer;
        raw_pointer get() const { return _where; }

    public:
        self_type& operator=(const self_type& p)
        {
            _where = p._where;
            return *this;
        }
        self_type& operator=(self_type&& p) noexcept
        {
            _where = p._where;
            return *this;
        }
        bool operator==(const self_type& p) const { return _where == p._where; }
        bool operator!=(const self_type& p) const { return !(*this == p); }
        reference operator*() const { return *_where; }
        // const_reference operator*() const { return *_where; };
        reference operator->() const { return *_where; }
        // const_reference operator->() const { return *_where; }
        self_type& operator++()
        {
            ++_where;
            return *this;
        };
        self_type operator++(int)
        {
            self_type out(_where);
            ++_where;
            return out;
        }
        self_type& operator--()
        {
            --_where;
            return *this;
        };
        self_type operator--(int)
        {
            self_type out(_where);
            --_where;
            return out;
        }

        self_type& operator+=(difference_type d)
        {
            _where += d;
            return *this;
        }
        self_type& operator-=(difference_type d)
        {
            _where -= d;
            return *this;
        }
        self_type operator+(difference_type d) const { return self_type(_where + d); }
        self_type operator-(difference_type d) const { return self_type(_where - d); }
        difference_type operator-(const self_type& p) const { return _where - p._where; }
        reference operator[](size_type i) const { return _where[i]; }
        // const_reference operator[](size_type i) const { return _where[i]; }
        bool operator<(const self_type& p) const { return _where < p._where; }
        bool operator<=(const self_type& p) const { return _where <= p._where; }
        bool operator>=(const self_type& p) const { return !(*this < p); }
        bool operator>(const self_type& p) const { return !(*this <= p); }
    };

    /*
     *   If RawType = false, no entry will be default-initialized/destructed and a memory copy may
     * be performed (even for initialization) (hence not touched at construction/resize)
     */
    template <class Type, bool RawType = false>
    class ContiguousStorage_
    {
    public:
        using self_type = ContiguousStorage_<Type, RawType>;
        template <bool Raw>
        using other_type = ContiguousStorage_<Type, Raw>;

        using pointer = ContiguousPointer<Type>;
        using const_pointer = ContiguousPointer<const Type>;
        using iterator = pointer;
        using const_iterator = const_pointer;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;

        using value_type = Type;
        using reference = Type&;
        using const_reference = const Type&;
        using size_type = typename pointer::size_type;
        using difference_type = typename pointer::difference_type;
        using raw_pointer = typename pointer::raw_pointer;

    private:
        mr::MemoryResource* _resource = 0;
        raw_pointer _pointer = 0;
        size_type _size = 0;
        size_type _capacity = 0;

    private:
        template <class ItType>
        using IsRandom = std::is_same<typename std::iterator_traits<ItType>::iterator_category,
                                      std::random_access_iterator_tag>;

        void _reserve_capacity(size_type count)
        {
            /* if capacity greater than count, return */

            // if (_capacity > 0)
            //                 _pointer = static_cast<value_type*>(
            //                     _resource->allocate(sizeof(value_type) * _capacity,
            //                     alignof(value_type)));
            //
        }
        void _reduce_capacity(size_type count)
        { /* must be equal to count (if zero, free ptr) */
            /* if count< than size, destruct all entries before reducing the size */
        }
        void _destruct_until(size_type target)
        {
            if constexpr (RawType)
                _size = target;
            else {
                while (_size > target)
                    _pointer[--_size].~value_type();
            }
        }

        /* expect memory to be consecutive (if count == 0, return) */
        void _content_copy(pointer to, pointer from, mr::MemoryResource* mrf, size_type count)
        { /* incomplete */
        }
        void _content_move(pointer to, pointer from, size_type count)
        { /* incomplete */
        }

    public:
        /* resource of null will take mr::defaultInstance */
        ContiguousStorage_(mr::MemoryResource* mr = 0)
        {
            if ((_resource = mr) == 0)
                _resource = mr::defaultInstance();
        }
        explicit ContiguousStorage_(size_type count, mr::MemoryResource* mr = 0)
        {
            if ((_resource = mr) == 0)
                _resource = mr::defaultInstance();
            _reserve_capacity(count);

            if constexpr (!RawType) {
                while (_size < count) {
                    new (_pointer + _size) value_type();
                    ++_size;
                }
            }
        }
        explicit ContiguousStorage_(size_type count, const_reference init,
                                    mr::MemoryResource* mr = 0)
        {
            if ((_resource = mr) == 0)
                _resource = mr::defaultInstance();
            _reserve_capacity(count);

            /* always initialize, no matter what RawType is, as an init-value has been given */
            while (_size < count)
                push_back(init);
        }
        template <class ItType>
        ContiguousStorage_(ItType ibegin, ItType iend, mr::MemoryResource* mr = 0)
        {
            if ((_resource = mr) == 0)
                _resource = mr::defaultInstance();

            /* check if we can reserve the capacity */
            if constexpr (IsRandom<ItType>::value)
                _reserve_capacity(iend - ibegin);

            /* always initialize, no matter what RawType is, as an init-value has been given */
            for (; ibegin != iend; ++ibegin)
                push_back(*ibegin);
        }
        ContiguousStorage_(std::initializer_list<value_type> l, mr::MemoryResource* mr = 0)
        {
            if ((_resource = mr) == 0)
                _resource = mr::defaultInstance();
            _reserve_capacity(l.size());

            /* always initialize, no matter what RawType is, as an init-value has been given */
            for (auto it = l.begin(); it != l.end(); ++it)
                push_back(*it);
        }

        /* resource of null will take s::resouce */
        template <bool Raw>
        ContiguousStorage_(const other_type<Raw>& s, mr::MemoryResource* mr = 0)
        {
            if ((_resource = mr) == 0)
                _resource = s._resource;
            _reserve_capacity(s._size);

            /* check if the data can just be copied */
            if constexpr (RawType && Raw)
                _content_copy(_pointer, s._size, s._resource, _size = s._size);
            else {
                while (_size < s._size)
                    push_back(s._pointer[_size]);
            }
        }
        template <bool Raw>
        ContiguousStorage_(other_type<Raw>&& s) noexcept
        {
            _resource = s._resource;
            std::swap(_pointer, s._pointer);
            std::swap(_size, s._size);
            std::swap(_capacity, s._capacity);
        }

        ~ContiguousStorage_() { _reduce_capacity(0); }

    public:
        mr::MemoryResource* resource() const { return _resource; }

        /* used by internal functions */
        void swap_resource(mr::MemoryResource* mr)
        {
            if (mr == _resource)
                return;
            if (_capacity == 0) {
                _resource = mr;
                return;
            }

            raw_pointer next = static_cast<raw_pointer>(
                mr->allocate(_capacity * sizeof(value_type), alignof(value_type)));
            _content_copy(next, _pointer, _resource, _size);
            std::swap(_pointer, next);
            std::swap(_resource, mr);

            mr->deallocate(next, _capacity * sizeof(value_type), alignof(value_type));
        }
        void push_back(const_reference value)
        {
            _reserve_capacity(_size + 1);
            new (_pointer + _size) value_type(value);
            ++_size;
        }
        void push_back(value_type&& value)
        {
            _reserve_capacity(_size + 1);
            new (_pointer + _size) value_type(std::move(value));
            ++_size;
        }

        /* incoming resource will be used */
        template <bool Raw>
        self_type& operator=(const other_type<Raw>& s)
        {
            assign(s, s._resource);
            return *this;
        }
        template <bool Raw>
        self_type& operator=(other_type<Raw>&& s)
        {
            std::swap(_resource, s._resource);
            std::swap(_pointer, s._pointer);
            std::swap(_size, s._size);
            std::swap(_capacity, s._capacity);
            s._reduce_capacity(0);
            return *this;
        }
        self_type& operator=(std::initializer_list<value_type> l)
        {
            assign(l, 0);
            return *this;
        }

        /* resource of null will keep the current resource */
        void assign(size_type count, const_reference init, mr::MemoryResource* mr = 0)
        {
            /* if the resource will be swapped, destruct beforehand, as this will
             *   prevent an additional unnecessary copying of the data */
            if (mr != 0) {
                _destruct_until(0);
                swap_resource(mr);
            }
            _reserve_capacity(count);

            for (size_type i = 0; i < _size; ++i)
                _pointer[i] = init;
            while (_size < count)
                push_back(init);

            _destruct_until(count);
        }
        template <class ItType>
        void assign(ItType ibegin, ItType iend, mr::MemoryResource* mr = 0)
        {
            /* if the resource will be swapped, destruct beforehand, as this will
             *   prevent an additional unnecessary copying of the data */
            if (mr != 0) {
                _destruct_until(0);
                swap_resource(mr);
            }

            /* check if we can reserve the capacity */
            if constexpr (IsRandom<ItType>::value)
                _reserve_capacity(iend - ibegin);

            size_type count = 0;
            while (ibegin != iend) {
                if (count < _size)
                    _pointer[count] = *ibegin;
                else
                    push_back(*ibegin);

                ++ibegin;
                ++count;
            }

            _destruct_until(count);
        }
        void assign(std::initializer_list<value_type> l, mr::MemoryResource* mr = 0)
        {
            /* if the resource will be swapped, destruct beforehand, as this will
             *   prevent an additional unnecessary copying of the data */
            if (mr != 0) {
                _destruct_until(0);
                swap_resource(mr);
            }
            _reserve_capacity(l.size());

            auto it = std::advance(l.begin(), std::min<size_type>(l.size(), _size));
            std::copy(l.begin(), it, _pointer);

            for (; it != l.end(); ++it)
                push_back(*it);
            _destruct_until(l.size());
        }
        template <bool Raw>
        void assign(const other_type<Raw>& s, mr::MemoryResource* mr = 0)
        {
            /* if the resource will be swapped, destruct beforehand, as this will
             *   prevent an additional unnecessary copying of the data */
            if (mr != 0) {
                _destruct_until(0);
                swap_resource(mr);
            }
            _reserve_capacity(s._size);

            /* check if the data can just be copied */
            if constexpr (RawType && Raw)
                _content_copy(_pointer, s._size, s._resource, _size = s._size);
            else {
                size_type off = std::min<size_type>(s._size, _size);
                std::copy(s._pointer, s._pointer + off, _pointer);

                while (off < s._size)
                    push_back(s._pointer[off++]);

                _destruct_until(s._size);
            }
        }

        reference at(size_type i)
        {
            if (i >= _size)
                throw std::out_of_range("Index into ContiguousStorage is out of range");
            return _pointer[i];
        }
        const_reference at(size_type i) const
        {
            if (i >= _size)
                throw std::out_of_range("Index into ContiguousStorage is out of range");
            return _pointer[i];
        }
        reference operator[](size_type i) { return _pointer[i]; }
        const_reference operator[](size_type i) const { return _pointer[i]; }
        reference front() { return *_pointer; }
        const_reference front() const { return *_pointer; }
        reference back() { return *(_pointer + _size - 1); }
        const_reference back() const { return *(_pointer + _size - 1); }
        pointer data() { return _pointer; }
        const_pointer data() const { return _pointer; }

        iterator begin() { return _pointer; }
        const_iterator begin() const { return _pointer; }
        const_iterator cbegin() const { return _pointer; }
        iterator end() { return (_pointer + _size); }
        const_iterator end() const { return (_pointer + _size); }
        const_iterator cend() const { return (_pointer + _size); }

        reverse_iterator rbegin() { return reverse_iterator(end()); }
        const_reverse_iterator rbegin() const { return const_reverse_iterator(cend()); }
        const_reverse_iterator crbegin() const { return const_reverse_iterator(cend()); }
        reverse_iterator rend() { return everse_iterator(begin()); }
        const_reverse_iterator rend() const { return const_reverse_iterator(cbegin()); }
        const_reverse_iterator crend() const { return const_reverse_iterator(cbegin()); }

        size_type size() const { return 0; }
        bool empty() const { return size() == 0; }
        size_type max_size() const { return std::numeric_limits<difference_type>::max(); }
        void reserve(size_type cap)
        { /* incomplete */
        }
        size_type capacity() const { return _capacity; }
        void shrink_to_fit()
        { /* incomplete */
        }

        void clear() noexcept
        { /* incomplete */
        }
        iterator insert(const_iterator i, const_reference value)
        { /* incomplete */
            return i;
        }
        iterator insert(const_iterator i, value_type&& value)
        { /* incomplete */
            return i;
        }
        iterator insert(const_iterator i, size_type count, const_reference value)
        { /* incomplete */
            return i;
        }
        template <class ItType>
        iterator insert(const_iterator i, ItType ibegin, ItType iend)
        { /* incomplete */
            return i;
        }
        iterator insert(const_iterator i, std::initializer_list<value_type> l)
        { /* incomplete */
            return i;
        }
        template <class... Args>
        iterator emplace(const_iterator i, Args&&... args)
        { /* incomplete */
            return i;
        }
        iterator erase(iterator i)
        { /* incomplete */
            return i;
        }
        iterator erase(const_iterator i)
        { /* incomplete */
            return i;
        }
        iterator erase(iterator ibegin, iterator iend)
        { /* incomplete */
            return ibegin;
        }
        iterator erase(const_iterator ibegin, const_iterator iend)
        { /* incomplete */
            return ibegin;
        }
        template <class... Args>
        reference emplace_back(Args&&... args)
        { /* incomplete */
            return *_pointer;
        }
        void pop_back()
        { /* incomplete */
        }
        void resize(size_type count)
        { /* incomplete */
        }
        void resize(size_type count, const_reference value)
        { /* incomplete */
        }
        template <bool Raw>
        void swap(other_type<Raw>& o) noexcept
        { /* incomplete */
        }
    };

    template <class T>
    // using ContiguousStorage = ContiguousStorage_<T>; //
    using ContiguousStorage = thrust::universal_vector<T>;

} // namespace elsa