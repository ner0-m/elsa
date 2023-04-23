#pragma once

#include "ContiguousMemory.h"

#include <iterator>
#include <limits>

namespace elsa::rm
{
    namespace type_tags
    {
        /*
         *  MemHandle: use memory-transfer functions to process copy/move
         *  UnInit: skip default initialization and destruction
         */
        template <bool MemHandle, bool UnInit>
        struct tag_template {
            static constexpr bool mem = MemHandle;
            static constexpr bool init = !UnInit;
        };

        struct complex : tag_template<false, false> {
        };
        struct trivial : tag_template<true, false> {
        };
        struct uninitialized : tag_template<true, true> {
        };
    } // namespace type_tags

    namespace detail
    {
        template <class Type>
        using is_random_iterator =
            std::is_same<typename std::iterator_traits<Type>::iterator_category,
                         std::random_access_iterator_tag>;

        /* check if Type defines a non-static member-function
         *   'base' and extract its return value */
        template <class Type>
        struct has_base_member {
            struct error_type {
            };

            template <class Tp, class RTp = decltype(std::declval<Tp>().base()),
                      class _enable = std::invoke_result_t<decltype(&Tp::base), Tp>>
            static RTp test(int);
            template <class Tp>
            static error_type test(...);

            using type = decltype(test<Type>(0));
            static constexpr bool has = !std::is_same<type, error_type>::value;
        };

        /* check if Type defines a non-static member-function
         *   'get' and extract its return value */
        template <class Type>
        struct has_get_member {
            struct error_type {
            };

            template <class Tp, class RTp = decltype(std::declval<Tp>().get()),
                      class _enable = std::invoke_result_t<decltype(&Tp::get), Tp>>
            static RTp test(int);
            template <class Tp>
            static error_type test(...);

            using type = decltype(test<Type>(0));
            static constexpr bool has = !std::is_same<type, error_type>::value;
        };

        template <class Type>
        using actual_pointer = std::is_pointer<std::remove_reference_t<Type>>;
        template <class Type>
        using base_returns_pointer =
            actual_pointer<std::conditional_t<has_base_member<Type>::has,
                                              typename has_base_member<Type>::type, void>>;
        template <class Type>
        using get_returns_pointer =
            actual_pointer<std::conditional_t<has_get_member<Type>::has,
                                              typename has_get_member<Type>::type, void>>;

        /*
         *  A container is a managing unit of 'capacity' number of types, allocated with the
         *  corresponding resource and managed with the given type-tags.
         *  - The first 'size' number of values are at all times considered constructed.
         *  - Size will always be less or equal to capacity.
         *  - A capacity of zero implies no allocation.
         *  - A pointer of zero is the null-container
         */
        template <class Type, class TypeTag>
        struct ContContainer {
        public:
            using self_type = ContContainer<Type, TypeTag>;
            using value_type = Type;
            using size_type = size_t;
            using difference_type = ptrdiff_t;

            using raw_pointer = Type*;
            using const_raw_pointer = const Type*;
            using reference = Type&;
            using const_reference = const Type&;

        public:
            mr::MemoryResource resource;
            raw_pointer pointer = 0;
            size_type size = 0;
            size_type capacity = 0;

        public:
            ContContainer() {}
            ContContainer(const mr::MemoryResource& r, raw_pointer p, size_type s, size_type c)
            {
                resource = r;
                pointer = p;
                size = s;
                capacity = c;
            }
            ContContainer(const self_type&) = delete;
            ContContainer(self_type&& c) noexcept
            {
                std::swap(resource, c.resource);
                std::swap(pointer, c.pointer);
                std::swap(size, c.size);
                std::swap(capacity, c.capacity);
            }
            ContContainer& operator=(const self_type&) = delete;
            ContContainer& operator=(self_type&& c) noexcept
            {
                std::swap(resource, c.resource);
                std::swap(pointer, c.pointer);
                std::swap(size, c.size);
                std::swap(capacity, c.capacity);
                return *this;
            }
            ~ContContainer()
            {
                if (pointer == 0)
                    return;
                destruct_until(0);
                resource->deallocate(pointer, capacity * sizeof(value_type), alignof(value_type));
            }

        public:
            raw_pointer end_ptr() const { return pointer + size; }
            void destruct_until(size_type count) noexcept
            {
                if constexpr (!TypeTag::init)
                    size = count;
                else {
                    while (size > count)
                        pointer[--size].~value_type();
                }
            }
            self_type swap(const mr::MemoryResource& mr, raw_pointer p, size_type sz, size_type cp)
            {
                self_type old{resource, pointer, size, capacity};
                resource = mr;
                pointer = p;
                size = sz;
                capacity = cp;
                return old;
            }

            /*
             *  The container will at most have capacity 'count' after the call
             *      -> must be larger than the current size
             */
            void reduce(size_type count) { reserve(count, count, resource); }

            /*
             *  count: how many to at least require
             *  move: if realloc, how many to move-construct over (must be less than size)
             *  mr: if not zero, force-relloc to the new memory resource
             *      (even if its the same resource)
             * returns null-container or old container on relocation in case of a
             * relocation, this container will have a size of 'copy'
             */
            self_type reserve(size_type count, size_type move,
                              const mr::MemoryResource& mr = mr::MemoryResource())
            {
                if (count <= capacity && !mr.valid())
                    return self_type();
                mr::MemoryResource new_mr = mr.valid() ? mr : resource;

                /* check if this is a noninitial reserving for a growing size in which
                 *   case the capacity should not just satisfy the request but by some form
                 *   of equation to prevent to many reallocations */
                size_type new_cap = std::max<size_type>(count, move);
                if (pointer != 0 && new_cap > capacity) {
                    size_type min_cap = new_cap;
                    new_cap = capacity;

                    while (new_cap < min_cap)
                        new_cap += (new_cap / 2);
                }

                /* check if an empty container has been requested */
                if (new_cap == 0)
                    return swap(new_mr, 0, 0, 0);

                /* check if a relocation could suffice (not considered a container-change) */
                if (pointer != 0 && new_mr == resource
                    && resource->tryResize(pointer, capacity * sizeof(value_type),
                                           alignof(value_type), new_cap * sizeof(value_type),
                                           alignof(value_type))) {
                    capacity = new_cap;
                    return self_type();
                }

                raw_pointer new_ptr = static_cast<raw_pointer>(
                    new_mr->allocate(new_cap * sizeof(value_type), alignof(value_type)));
                self_type old = swap(new_mr, new_ptr, 0, new_cap);

                /* either move-initialize the data or perform the memory move */
                if constexpr (!TypeTag::mem) {
                    for (size_type i = 0; i < move; ++i) {
                        new (pointer + i) value_type(std::move(old.pointer[i]));
                        ++size;
                    }
                } else if (move > 0) {
                    new_mr->copyMemory(new_ptr, old.pointer, move * sizeof(value_type));
                    size = move;
                }
                return old;
            }

            /* move [where; end] by diff within the container */
            void move_tail(size_type where, difference_type diff)
            {
                /* check if this move can be delegated to the mr */
                if constexpr (TypeTag::mem) {
                    resource->moveMemory(pointer + (where + diff), pointer + where,
                                         (size - where) * sizeof(value_type));
                    size = std::max<size_type>(size, (size - where) + diff);
                    return;
                }

                /* check if the entire content needs to be moved down (i.e. size decrease) */
                if (diff <= 0) {
                    /* reminder: diff is negative */
                    while (where < size) {
                        pointer[where + diff] = std::move(pointer[where]);
                        ++where;
                    }
                    destruct_until(size + diff);
                    return;
                }

                /* first move the upper 'new' part in order to ensure the entire buffer is
                 *   constructed and an exception would therefore not leave an invalid state */
                size_type end = size, next = size - diff;
                while (next < end) {
                    new (pointer + size) value_type(std::move(pointer + next));
                    ++size;
                    ++next;
                }

                /* move the remaining part up into the already, but moved out slots */
                end = where + diff;
                while (--next >= end)
                    pointer[next] = std::move(pointer[next - diff]);
            }

            /* copy a range into a given range */
            template <class ItType>
            void insert_range(size_type off, ItType ibegin, ItType iend, size_type count)
            {
                size_type end = off + count;

                /* check if the iterator has a 'base' function to get the raw pointer */
                if constexpr (detail::base_returns_pointer<ItType>::value
                              && detail::is_random_iterator<ItType>::value)
                    insert_range(off, ibegin.base(), iend.base(), count);

                /* check if the iterator has a 'get' function to get the raw pointer */
                else if constexpr (detail::get_returns_pointer<ItType>::value
                                   && detail::is_random_iterator<ItType>::value)
                    insert_range(off, ibegin.get(), iend.get(), count);

                /* check if the iterator is a pointer and can be delegated to mr */
                else if constexpr (detail::actual_pointer<ItType>::value && TypeTag::mem) {
                    resource->copyMemory(pointer + off, ibegin, count * sizeof(value_type));
                    size = std::max<size_type>(size, end);
                }

                /* insert the range by using the iterators */
                else {
                    size_type transition = std::min<size_type>(size, end);

                    /* copy-assign the currently existing and to-be-overwritten part */
                    while (off < transition) {
                        pointer[off] = *ibegin;
                        ++off;
                        ++ibegin;
                    }

                    /* copy-construct the remaining part */
                    while (ibegin != iend) {
                        new (pointer + size) value_type(*ibegin);
                        ++size;
                        ++ibegin;
                    }
                }
            }

            /* write same value (or default-construct/reset) into a given range */
            void set_range(size_type off, const_raw_pointer value, size_type count)
            {
                size_type end = off + count;
                size_type transition = std::min<size_type>(size, end);

                /* set/update the range with the default constructor/value */
                if (value == 0) {
                    /* check if the setting can be entirely skipped or delegated to the mr */
                    if constexpr (!TypeTag::init)
                        size = std::max<size_type>(size, end);
                    else if constexpr (TypeTag::mem) {
                        value_type tmp;
                        resource->setMemory(pointer + off, &tmp, sizeof(value_type), count);
                        size = std::max<size_type>(size, end);

                    } else {
                        while (off < transition)
                            pointer[off++] = std::move(value_type());

                        while (size < end) {
                            new (pointer + size) value_type();
                            ++size;
                        }
                    }
                }

                /* set/update the range with the given value */
                else if constexpr (TypeTag::mem) {
                    resource->setMemory(pointer + off, value, sizeof(value_type), count);
                    size = std::max<size_type>(size, end);
                }

                else {
                    while (off < transition)
                        pointer[off++] = *value;

                    while (size < end) {
                        new (pointer + size) value_type(*value);
                        ++size;
                    }
                }
            }
        };

        /*
         *  Used Both as Pointer Wrapper and Iterator (hence the iterator_category)
         *      -> Necessary for integration with thrust helper types
         */
        template <class Type>
        class ContPointer
        {
        public:
            using self_type = ContPointer<Type>;
            using value_type = std::remove_cv_t<Type>;
            using size_type = size_t;
            using difference_type = ptrdiff_t;
            using iterator_category = std::random_access_iterator_tag;

            using raw_pointer = Type*;
            using pointer = Type*;
            using reference = Type&;

        private:
            pointer _where = 0;

        public:
            ContPointer() {}
            ContPointer(pointer w) : _where(w) {}
            ContPointer(const self_type& p) : _where(p._where) {}
            ContPointer(self_type&& p) noexcept : _where(p._where) {}

        public:
            /*
             *  for compatability with thrust
             */
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
            pointer operator->() const { return _where; }
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
            raw_pointer base() const { return _where; }
        };

    } // namespace detail

    /*
     *  ContiguousStorage behaves like stl-container except for in the exception case.
     *    - Exceptions by type-specific functions: the object will remain in a valid state after the
     *      exception, might however be in a modified state (i.e. partially inserted...)
     *    - Exceptions by the allocator will result in this container unmodified or
     *      the operation fully performed.
     *
     *  Each ContiguousStorage is associated with a rm::MemoryResource (polymorphic allocators).
     *  All allocations and memory-operations are performed on the currently used resource.
     *  Changes to the current resource associated with this container:
     *    - copy-construct (optionally inherit the incoming resource)
     *    - move-construct (inherit the incoming resource)
     *    - other-constructors (parameter-resource or rm::defaultInstance)
     *    - swap_resource calls
     *    - copy-assignment or move-assignment operator (will inherit the incoming resource)
     *    - assign()/assign_default() calls (optional parameter-resource else unchanged)
     *    - swap() will swap resource with other container
     */
    template <class Type, class TypeTag = type_tags::complex>
    class ContiguousStorage
    {
    public:
        using self_type = ContiguousStorage<Type, TypeTag>;
        using container_type = detail::ContContainer<Type, TypeTag>;
        using value_type = typename container_type::value_type;
        using size_type = typename container_type::size_type;
        using difference_type = typename container_type::difference_type;

        using raw_pointer = typename container_type::raw_pointer;
        using const_raw_pointer = typename container_type::const_raw_pointer;
        using reference = typename container_type::reference;
        using const_reference = typename container_type::const_reference;

        using pointer = detail::ContPointer<Type>;
        using const_pointer = detail::ContPointer<const Type>;
        using iterator = pointer;
        using const_iterator = const_pointer;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    private:
        container_type _self;

    public:
        /* invalid resource will take mr::defaultInstance */
        ContiguousStorage(const mr::MemoryResource& mr = mr::MemoryResource())
        {
            if (!(_self.resource = mr).valid())
                _self.resource = mr::defaultInstance();
        }
        explicit ContiguousStorage(size_type count,
                                   const mr::MemoryResource& mr = mr::MemoryResource())
        {
            if (!(_self.resource = mr).valid())
                _self.resource = mr::defaultInstance();

            _self.reserve(count, 0);
            _self.set_range(0, 0, count);
        }
        explicit ContiguousStorage(size_type count, const_reference init,
                                   const mr::MemoryResource& mr = mr::MemoryResource())
        {
            if (!(_self.resource = mr).valid())
                _self.resource = mr::defaultInstance();

            _self.reserve(count, 0);
            _self.set_range(0, &init, count);
        }
        template <class ItType>
        ContiguousStorage(ItType ibegin, ItType iend,
                          const mr::MemoryResource& mr = mr::MemoryResource())
        {
            if (!(_self.resource = mr).valid())
                _self.resource = mr::defaultInstance();

            size_type count = std::distance(ibegin, iend);
            _self.reserve(count, 0);
            _self.insert_range(0, ibegin, iend, count);
        }
        ContiguousStorage(std::initializer_list<value_type> l,
                          const mr::MemoryResource& mr = mr::MemoryResource())
        {
            if (!(_self.resource = mr).valid())
                _self.resource = mr::defaultInstance();

            _self.reserve(l.size(), 0);
            _self.insert_range(0, l.begin(), l.end(), l.size());
        }

        /* invalid resource will take s::resouce */
        ContiguousStorage(const self_type& s, const mr::MemoryResource& mr = mr::MemoryResource())
        {
            if (!(_self.resource = mr).valid())
                _self.resource = s._self.resource;

            _self.reserve(s._self.size, 0);
            _self.insert_range(0, s._self.pointer, s._self.end_ptr(), s._self.size);
        }
        ContiguousStorage(self_type&& s) noexcept { std::swap(_self, s._self); }

        ~ContiguousStorage() = default;

    public:
        /* invalid resource will take mr::defaultInstance */
        mr::MemoryResource resource() const { return _self.resource; }
        void swap_resource(const mr::MemoryResource& mr)
        {
            mr::MemoryResource actual = mr.valid() ? mr : mr::defaultInstance();
            if (actual == _self.resource)
                return;
            _self.reserve(0, _self.size, actual);
        }

        /* incoming resource will be used */
        self_type& operator=(const self_type& s)
        {
            if (&s == this)
                return *this;
            if (s._self.resource == _self.resource)
                assign(s);
            else {
                _self.reserve(s._self.size, 0, s._self.resource);
                _self.insert_range(0, s._self.pointer, s._self.end_ptr(), s._self.size);
            }
            return *this;
        }
        self_type& operator=(self_type&& s) noexcept
        {
            if (&s == this)
                return *this;
            std::swap(_self, s._self);
            s._self.destruct_until(0);
            return *this;
        }
        self_type& operator=(std::initializer_list<value_type> l)
        {
            assign(l);
            return *this;
        }

        /* current resource will be used */
        void assign_default(size_type count, const mr::MemoryResource& mr = mr::MemoryResource())
        {
            _self.reserve(count, 0, mr);
            _self.set_range(0, 0, count);
            _self.destruct_until(count);
        }
        void assign(size_type count, const_reference init,
                    const mr::MemoryResource& mr = mr::MemoryResource())
        {
            _self.reserve(count, 0, mr);
            _self.set_range(0, &init, count);
            _self.destruct_until(count);
        }
        template <class ItType>
        void assign(ItType ibegin, ItType iend, const mr::MemoryResource& mr = mr::MemoryResource())
        {
            size_type count = std::distance(ibegin, iend);
            _self.reserve(count, 0, mr);
            _self.insert_range(0, ibegin, iend, count);
            _self.destruct_until(count);
        }
        void assign(std::initializer_list<value_type> l,
                    const mr::MemoryResource& mr = mr::MemoryResource())
        {
            assign(l.begin(), l.end(), mr);
        }
        void assign(const self_type& s, const mr::MemoryResource& mr = mr::MemoryResource())
        {
            assign(s._self.pointer, s._self.end_ptr(), mr);
        }

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
        reference back() { return *(_self.end_ptr() - 1); }
        const_reference back() const { return *(_self.end_ptr() - 1); }
        pointer data() { return _self.pointer; }
        const_pointer data() const { return _self.pointer; }

        iterator begin() { return _self.pointer; }
        const_iterator begin() const { return _self.pointer; }
        const_iterator cbegin() const { return _self.pointer; }
        iterator end() { return (_self.end_ptr()); }
        const_iterator end() const { return (_self.end_ptr()); }
        const_iterator cend() const { return (_self.end_ptr()); }

        reverse_iterator rbegin() { return reverse_iterator(end()); }
        const_reverse_iterator rbegin() const { return const_reverse_iterator(cend()); }
        const_reverse_iterator crbegin() const { return const_reverse_iterator(cend()); }
        reverse_iterator rend() { return everse_iterator(begin()); }
        const_reverse_iterator rend() const { return const_reverse_iterator(cbegin()); }
        const_reverse_iterator crend() const { return const_reverse_iterator(cbegin()); }

        size_type size() const { return _self.size; }
        bool empty() const { return size() == 0; }
        size_type max_size() const { return std::numeric_limits<difference_type>::max(); }
        void reserve(size_type cap) { _self.reserve(cap, _self.size); }
        size_type capacity() const { return _self.capacity; }
        void shrink_to_fit() { _self.reduce(_self.size); }

        void clear() noexcept { _self.destruct_until(0); }
        iterator insert_default(const_iterator i, size_type count)
        {
            difference_type off = i - _self.pointer;
            container_type old = _self.reserve(_self.size + count, off);

            if (old.pointer == 0)
                _self.move_tail(_self.pointer + off, count);
            _self.set_range(off, 0, count);
            if (old.pointer != 0)
                _self.insert_range(_self.size, old.pointer + off, old.end_ptr(), old.size - off);
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
            container_type old = _self.reserve(_self.size + count, off);

            if (old.pointer == 0)
                _self.move_tail(_self.pointer + off, count);
            _self.set_range(off, &value, count);
            if (old.pointer != 0)
                _self.insert_range(_self.size, old.pointer + off, old.end_ptr(), old.size - off);
            return _self.pointer + off;
        }
        template <class ItType>
        iterator insert(const_iterator i, ItType ibegin, ItType iend)
        {
            difference_type off = i - _self.pointer;
            difference_type total = std::distance(ibegin, iend);
            container_type old = _self.reserve(_self.size + total, off);

            if (old.pointer == 0)
                _self.move_tail(_self.pointer + off, off);
            _self.insert_range(off, ibegin, iend, total);
            if (old.pointer != 0)
                _self.insert_range(_self.size, old.pointer + off, old.end_ptr(), old.size - off);

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
            container_type old = _self.reserve(_self.size + 1, off);

            if (old.pointer == 0) {
                _self.move_tail(_self.pointer + off, 1);
                _self.pointer[off] = std::move(value_type(std::forward<Args>(args)...));
            } else {
                new (_self.end_ptr()) value_type(std::forward<Args>(args)...);
                _self.insert_range(++_self.size, old.pointer + off, old.end_ptr(), old.size - off);
            }
            return _self.pointer + off;
        }
        iterator erase(iterator i) { return erase(i, std::next(i)); }
        iterator erase(const_iterator i) { return erase(i, std::next(i)); }
        iterator erase(iterator ibegin, iterator iend)
        {
            return erase(const_iterator(ibegin.get()), const_iterator(iend.get()));
        }
        iterator erase(const_iterator ibegin, const_iterator iend)
        {
            difference_type offset = ibegin - _self.pointer;
            difference_type count = iend - ibegin;
            _self.move_tail(iend - _self.pointer, -count);
            return _self.pointer + offset;
        }
        template <class... Args>
        reference emplace_back(Args&&... args)
        {
            _self.reserve(_self.size + 1, _self.size);
            new (_self.end_ptr()) value_type(std::forward<Args>(args)...);
            return _self.pointer[_self.size++];
        }
        void push_back(const_reference value) { emplace_back(value); }
        void push_back(value_type&& value) { emplace_back(std::move(value)); }
        void pop_back() { _self.destruct_until(_self.size - 1); }
        void resize(size_type count)
        {
            if (count <= _self.size) {
                _self.destruct_until(count);
                return;
            }
            _self.reserve(count, _self.size);
            _self.set_range(_self.size, 0, count - _self.size);
        }
        void resize(size_type count, const_reference value)
        {
            if (count <= _self.size) {
                _self.destruct_until(count);
                return;
            }
            _self.reserve(count, _self.size);
            _self.set_range(_self.size, &value, count - _self.size);
        }
        void swap(self_type& o) noexcept { std::swap(_self, o._self); }
    };
} // namespace elsa::rm
