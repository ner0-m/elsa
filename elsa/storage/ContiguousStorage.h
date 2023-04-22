#pragma once

#include "memory_resource/ContiguousMemory.h"

#include <iterator>

#include "DisableWarnings.h"

DISABLE_WARNING_PUSH
DISABLE_WARNING_SIGN_CONVERSION
// Thrust is smart enough to always pick the correct vector for us
#include <thrust/universal_vector.h>
DISABLE_WARNING_POP

/*
 *  if iterator is pointer, it must be continuous, hence optimizations can be made
 *  implement MemoryResource reference counting!
 */

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
        reference operator->() const { return *_where; }
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
        bool operator<(const self_type& p) const { return _where < p._where; }
        bool operator<=(const self_type& p) const { return _where <= p._where; }
        bool operator>=(const self_type& p) const { return !(*this < p); }
        bool operator>(const self_type& p) const { return !(*this <= p); }
    };

    /*
     *   If RawType = true, no entry will be default-initialized/destructed and a memory copy may
     *      be performed (even for initialization) (hence not touched at construction/resize)
     *
     *  Behaves like stl-container except for in the exception case.
     *  The object will remain in a valid state after the exception, might however be in a modified
     *      state (i.e. partially inserted...)
     */
    template <class Type, bool RawType = false>
    class ContiguousStorage_
    {
    public:
        using self_type = ContiguousStorage_<Type, RawType>;

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
        struct _container {
        public:
            mr::MemoryResource* resource = 0;
            raw_pointer pointer = 0;
            size_type size = 0;
            size_type capacity = 0;

        public:
            _container() {}
            _container(const _container&) = delete;
            _container(_container&& c) noexcept
            {
                std::swap(resource, c.resource);
                std::swap(pointer, c.pointer);
                std::swap(size, c.size);
                std::swap(capacity, c.capacity);
            }
            _container& operator=(const _container&) = delete;
            _container& operator=(_container&& c) noexcept
            {
                std::swap(resource, c.resource);
                std::swap(pointer, c.pointer);
                std::swap(size, c.size);
                std::swap(capacity, c.capacity);
                return *this;
            }
            ~_container()
            {
                if (pointer != 0) {
                    destruct_until(0);
                    resource->deallocate(pointer, capacity * sizeof(value_type),
                                         alignof(value_type));
                }
            }
            void release()
            {
                if (pointer == 0)
                    return;
                destruct_until(0);
                resource->deallocate(pointer, capacity * sizeof(value_type), alignof(value_type));
                pointer = 0;
            }
            void destruct_until(size_type count) noexcept
            {
                if constexpr (RawType)
                    size = count;
                else {
                    while (size > count)
                        pointer[--size].~value_type();
                }
            }

            /*
             *   count: how many to at least require
             *   copy: if realloc, how many to copy-construct over
             *   mr: if not zero, force-relloc to the new memory resource
             *   returns null-container or old container on relocation
             *   in case of a relocation, this container will have a size of 'copy'
             */
            _container reserve(size_type count, size_type copy, mr::MemoryResource* mr = 0)
            { /* incomplete */
                return _container();
            }

            void reduce(size_type count)
            {
                /* must be equal to count (if zero, free ptr) */
                /* if count< than size, destruct all entries before reducing the size */
            }
        };

    private:
        _container _self;

    private:
        template <class ItType>
        using _is_random = std::is_same<typename std::iterator_traits<ItType>::iterator_category,
                                        std::random_access_iterator_tag>;
        /*
         *   move [where; end] by diff within the container
         *      destructs remainder on negative moves
         *      expects container to be large enough for the movement
         */
        void _self_move_tail(size_type where, difference_type diff)
        {
            /* check if content is moved up (i.e. the size increased) */
            if (diff > 0) {
                /* first move the upper 'new' part in order to ensure the entire buffer is
                 *   constructed and an exception would therefore not leave an invalid state */
                size_type end = _self.size, next = _self.size - diff;
                while (next < end) {
                    new (_self.pointer + _self.size) value_type(std::move(_self.pointer + next));
                    ++_self.size;
                    ++next;
                }

                /* move the remaining part up into the already, but moved out slots */
                end = where + diff;
                while (--next >= end)
                    _self.pointer[next] = std::move(_self.pointer[next - diff]);
                return;
            }

            /* move the entire content down and afterwards destruct the tail
             *   (mind: diff is negative!) */
            while (where < _self.size) {
                _self.pointer[where + diff] = std::move(_self.pointer[where]);
                ++where;
            }
            _self.destruct_until(where + diff);
        }

        /*
         *   copy a range into a given range (potentially resize)
         *      reserves necessary capacity
         */
        template <class ItType>
        void _copy_update_construct(size_type off, ItType ibegin, ItType iend, size_type count)
        {
            /* copy-assign the currently existing and to-be-overwritten part */
            _self.reserve(off + count, off);
            size_type transition = std::min<size_type>(_self.size, off + count);
            while (off < transition) {
                _self.pointer[off] = *ibegin;
                ++off;
                ++ibegin;
            }

            /* copy-construct the remaining part */
            while (ibegin != iend) {
                new (_self.pointer + _self.size) value_type(*ibegin);
                ++_self.size;
                ++ibegin;
            }
        }

        /*
         *   write same value (or default-construct/reset) into a given range (potentially resize)
         *      reserves necessary capacity
         */
        template <bool DefConstruct>
        void _set_update_construct(size_type off, const_pointer val, size_type count)
        {
            _self.reserve(off + count, off);

            size_type end = off + count;
            size_type transition = std::min<size_type>(_self.size, end);

            if constexpr (DefConstruct) {
                /* move-assign the still constructed values */
                while (off < transition)
                    _self.pointer[off++] = std::move(value_type());

                /* default-construct the new values */
                while (_self.size < end) {
                    new (_self.pointer + _self.size) value_type();
                    ++_self.size;
                }
            }

            else {
                /* copy-assign the still constructed values */
                while (off < transition)
                    _self.pointer[off++] = *val;

                /* copy-construct the new values */
                while (_self.size < end) {
                    new (_self.pointer + _self.size) value_type(*val);
                    ++_self.size;
                }
            }
        }

    public:
        /* resource of null will take mr::defaultInstance */
        ContiguousStorage_(mr::MemoryResource* mr = 0)
        {
            if ((_self.resource = mr) == 0)
                _self.resource = mr::defaultInstance();
        }
        explicit ContiguousStorage_(size_type count, mr::MemoryResource* mr = 0)
        {
            if ((_self.resource = mr) == 0)
                _self.resource = mr::defaultInstance();

            _set_update_construct<true>(0, 0, count);
        }
        explicit ContiguousStorage_(size_type count, const_reference init,
                                    mr::MemoryResource* mr = 0)
        {
            if ((_self.resource = mr) == 0)
                _self.resource = mr::defaultInstance();

            _set_update_construct<false>(0, &init, count);
        }
        template <class ItType>
        ContiguousStorage_(ItType ibegin, ItType iend, mr::MemoryResource* mr = 0)
        {
            if ((_self.resource = mr) == 0)
                _self.resource = mr::defaultInstance();

            _copy_update_construct(0, ibegin, iend, std::distance(ibegin, iend));
        }
        ContiguousStorage_(std::initializer_list<value_type> l, mr::MemoryResource* mr = 0)
        {
            if ((_self.resource = mr) == 0)
                _self.resource = mr::defaultInstance();

            _copy_update_construct(0, l.begin(), l.end(), l.size());
        }

        /* resource of null will take s::resouce */
        ContiguousStorage_(const self_type& s, mr::MemoryResource* mr = 0)
        {
            if ((_self.resource = mr) == 0)
                _self.resource = s._self.resource;

            _copy_update_construct(0, s._self.pointer, s._self.pointer + s._self.size,
                                   s._self.size);
        }
        ContiguousStorage_(self_type&& s) noexcept { std::swap(_self, s._self); }

        ~ContiguousStorage_() { _self.reduce(0); }

    public:
        mr::MemoryResource* resource() const { return _self.resource; }
        void swap_resource(mr::MemoryResource* mr)
        {
            if (mr == _self.resource)
                return;
            if (_self.capacity != 0)
                _self.reserve(0, _self.size, mr);
            else
                _self.resource = mr;
        }

        /* incoming resource will be used */
        self_type& operator=(const self_type& s)
        {
            if (s._self.resource == _self.resource)
                assign(s);
            else {
                _self.reserve(s._self.size, 0, s._self.resource);
                _copy_update_construct(0, s._self.pointer, s._self.pointer + s._self.size,
                                       s._self.size);
            }
            return *this;
        }
        self_type& operator=(self_type&& s)
        {
            std::swap(_self, s._self);
            s._self.reduce(0);
            return *this;
        }
        self_type& operator=(std::initializer_list<value_type> l)
        {
            assign(l);
            return *this;
        }

        /* current resource will be used */
        void assign_default(size_type count)
        {
            _self.reserve(count, 0);
            _set_update_construct<true>(0, 0, count);
            _self.destruct_until(count);
        }
        void assign(size_type count, const_reference init)
        {
            _self.reserve(count, 0);
            _set_update_construct<false>(0, &init, count);
            _self.destruct_until(count);
        }
        template <class ItType>
        void assign(ItType ibegin, ItType iend)
        {
            size_type count = std::distance(ibegin, iend);
            _copy_update_construct(0, ibegin, iend, count);
            _self.destruct_until(count);
        }
        void assign(std::initializer_list<value_type> l) { assign(l.begin(), l.end()); }
        void assign(const self_type& s) { assign(s._self.pointer, s._self.pointer + s._self.size); }

        reference at(size_type i)
        {
            if (i >= _self.size)
                throw std::out_of_range("Index into ContiguousStorage is out of range");
            return _self.pointer[i];
        }
        const_reference at(size_type i) const
        {
            if (i >= _self.size)
                throw std::out_of_range("Index into ContiguousStorage is out of range");
            return _self.pointer[i];
        }
        reference operator[](size_type i) { return _self.pointer[i]; }
        const_reference operator[](size_type i) const { return _self.pointer[i]; }
        reference front() { return *_self.pointer; }
        const_reference front() const { return *_self.pointer; }
        reference back() { return *(_self.pointer + _self.size - 1); }
        const_reference back() const { return *(_self.pointer + _self.size - 1); }
        pointer data() { return _self.pointer; }
        const_pointer data() const { return _self.pointer; }

        iterator begin() { return _self.pointer; }
        const_iterator begin() const { return _self.pointer; }
        const_iterator cbegin() const { return _self.pointer; }
        iterator end() { return (_self.pointer + _self.size); }
        const_iterator end() const { return (_self.pointer + _self.size); }
        const_iterator cend() const { return (_self.pointer + _self.size); }

        reverse_iterator rbegin() { return reverse_iterator(end()); }
        const_reverse_iterator rbegin() const { return const_reverse_iterator(cend()); }
        const_reverse_iterator crbegin() const { return const_reverse_iterator(cend()); }
        reverse_iterator rend() { return everse_iterator(begin()); }
        const_reverse_iterator rend() const { return const_reverse_iterator(cbegin()); }
        const_reverse_iterator crend() const { return const_reverse_iterator(cbegin()); }

        size_type size() const { return 0; }
        bool empty() const { return size() == 0; }
        size_type max_size() const { return std::numeric_limits<difference_type>::max(); }
        void reserve(size_type cap) { _self.reserve(cap, _self.size); }
        size_type capacity() const { return _self.capacity; }
        void shrink_to_fit() { _self.reduce(_self.size); }

        void clear() noexcept { _self.destruct_until(0); }
        iterator insert_default(const_iterator i, size_type count)
        {
            difference_type off = i - _self.pointer;
            _container old = _self.reserve(_self.size + count, off);

            if (old.pointer == 0)
                _self_move_tail(_self.pointer + off, count);
            _set_update_construct<true>(off, 0, count);
            if (old.pointer != 0)
                _copy_update_construct(_self.size, old.pointer + off, old.pointer + old.size,
                                       old.size - off);
            return _self.pointer + off;
        }
        iterator insert(const_iterator i, const_reference value) { return emplace(i, value); }
        iterator insert(const_iterator i, value_type&& value)
        {
            return emplace(i, std::move(value));
        }
        iterator insert(const_iterator i, size_type count, const_reference value)
        {
            difference_type off = i - _self.pointer;
            _container old = _self.reserve(_self.size + count, off);

            if (old.pointer == 0)
                _self_move_tail(_self.pointer + off, count);
            _set_update_construct<false>(off, &value, count);
            if (old.pointer != 0)
                _copy_update_construct(_self.size, old.pointer + off, old.pointer + old.size,
                                       old.size - off);
            return _self.pointer + off;
        }
        template <class ItType>
        iterator insert(const_iterator i, ItType ibegin, ItType iend)
        {
            difference_type off = i - _self.pointer;
            difference_type total = std::distance(ibegin, iend);
            _container old = _self.reserve(_self.size + total, off);

            if (old.pointer == 0)
                _self_move_tail(_self.pointer + off, off);
            _copy_update_construct(off, ibegin, iend, total);
            if (old.pointer != 0)
                _copy_update_construct(_self.size, old.pointer + off, old.pointer + old.size,
                                       old.size - off);

            return _self.pointer + off;
        }
        iterator insert(const_iterator i, std::initializer_list<value_type> l)
        {
            return insert(i, l.begin(), l.end());
        }
        template <class... Args>
        iterator emplace(const_iterator i, Args&&... args)
        {
            difference_type off = i - _self.pointer;
            _container old = _self.reserve(_self.size + 1, off);

            if (old.pointer == 0) {
                _self_move_tail(_self.pointer + off, 1);
                _self.pointer[off] = std::move(value_type(std::forward<Args>(args)...));
            } else {
                new (_self.pointer + _self.size) value_type(std::forward<Args>(args)...);
                _copy_update_construct(++_self.size, old.pointer + off, old.pointer + old.size,
                                       old.size - off);
            }
            return _self.pointer + off;
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
        {
            _self.reserve(_self.size + 1);
            new (_self.pointer + _self.size) value_type(std::forward<Args>(args)...);
            return _self.pointer[_self.size++];
        }
        void push_back(const_reference value) { emplace_back(value); }
        void push_back(value_type&& value) { emplace_back(std::move(value)); }
        void pop_back()
        { /* incomplete */
        }
        void resize(size_type count)
        { /* incomplete */
        }
        void resize(size_type count, const_reference value)
        { /* incomplete */
        }
        void swap(self_type& o) noexcept
        { /* incomplete */
        }
    };

    template <class T>
    // using ContiguousStorage = ContiguousStorage_<T>;
    using ContiguousStorage = thrust::universal_vector<T>;

} // namespace elsa