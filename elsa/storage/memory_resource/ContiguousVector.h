#pragma once

#include "MemoryResource.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <cstring>
#include <functional>
#include <atomic>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/universal_vector.h>
#include <thrust/execution_policy.h>

namespace elsa::mr
{
    namespace type_tags
    {
        /* - usual type handling as expected */
        struct complex {
        };

        /* - use memory-transfer functions to process copy/move/set of larger ranges
         * - single values or non-continuous iterators may still use assigning/construction
         * - default-initialization will invoke the default constructor once
         * - no destruction performed */
        struct trivial {
        };

        /* - additionally to trivial, no default construction is ever performed
         * - other constructors may still be called */
        struct uninitialized : public trivial {
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

        template <class Tag>
        constexpr bool is_trivial = std::is_base_of<type_tags::trivial, Tag>::value;

        template <class Tag>
        constexpr bool is_uninitialized = std::is_base_of<type_tags::uninitialized, Tag>::value;

        template <class Tag>
        constexpr bool is_complex = std::is_base_of<type_tags::complex, Tag>::value;

        /* type-wrapper for the pointer to be casted to to prevent native
         *  opeartions on the type to be called (as the contigous-vector already takes care of them)
         */
        template <class Type>
        struct type_wrapper {
            uint8_t payload[sizeof(Type)];
        };

        template <class Type>
        void fillMemory(Type* ptr, const void* src, size_t count)
        {
            using Tp = type_wrapper<Type>;

            auto src_ = reinterpret_cast<const Tp*>(src);
            auto dst_ = thrust::universal_ptr<Tp>(reinterpret_cast<Tp*>(ptr));

            thrust::fill(thrust::device, dst_, dst_ + count, *src_);
        }
        template <class Type>
        void copyMemory(void* ptr, const void* src, size_t count, bool src_universal)
        {
            using Tp = type_wrapper<Type>;

            auto src_ = reinterpret_cast<const Tp*>(src);
            auto dst_ = reinterpret_cast<Tp*>(ptr);

            if (src_universal) {
                auto usrc = thrust::universal_ptr<const Tp>(src_);
                auto udst = thrust::universal_ptr<Tp>(dst_);
                thrust::copy(thrust::device, usrc, usrc + count, udst);
            } else
                thrust::copy(thrust::host, src_, src_ + count, dst_);
        }
        template <class Type>
        void moveMemory(void* ptr, const void* src, size_t count)
        {
            std::memmove(ptr, src, count * sizeof(Type));
        }

        /*
         *  A container is a reference-counted managing unit of 'capacity' number of types,
         *  allocated with the corresponding resource and managed with the given type-tags.
         *  - The first 'size' number of values are at all times considered constructed.
         *  - Size will always be less or equal to capacity.
         *  - A capacity of zero implies no allocation and therefore a null-pointer (vice versa).
         *  - A pointer of zero is the null-container.
         *
         *  Additionally the container can also manage external alloctions. For
         *  this, it will call the cleanup function whenever the container is not used anymore,
         *  and resizing is disabled.
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
            raw_pointer pointer = nullptr;
            size_type size = 0;
            size_type capacity = 0;
            std::function<void()> release;

        public:
            ContContainer()
            {
                static_assert(
                    is_trivial<TypeTag> || is_uninitialized<TypeTag> || is_complex<TypeTag>,
                    "unknown type-tag encountered (only complex, trivial, uninitialized known)");
            }

            ContContainer(const mr::MemoryResource& r, raw_pointer p, size_type s, size_type c)
            {
                resource = r;
                pointer = p;
                size = s;
                capacity = c;
            }

            ContContainer(const mr::MemoryResource& r, raw_pointer p, size_type s,
                          std::function<void()>&& cleanup)
            {
                resource = r;
                pointer = p;
                size = s;
                capacity = s;
                release = std::move(cleanup);
            }

            ContContainer(const self_type&) = delete;

            ContContainer(self_type&& c) = delete;

            ContContainer& operator=(const self_type&) = delete;

            ContContainer& operator=(self_type&& c) = delete;

            ~ContContainer()
            {
                if (release) {
                    release();
                    return;
                }
                if (pointer == nullptr)
                    return;
                destruct_until(0);
                resource->deallocate(pointer, capacity * sizeof(value_type), alignof(value_type));
            }

        public:
            raw_pointer end_ptr() const { return pointer + size; }

            void destruct_until(size_type count) noexcept
            {
                if constexpr (is_trivial<TypeTag>)
                    size = count;
                else {
                    while (size > count)
                        pointer[--size].~value_type();
                }
            }

            /*
             *  The container will at most have capacity 'count' after the call
             *      -> count must be smaller or equal to 'size'
             */
            std::shared_ptr<self_type> reduce(size_type count, bool single_ref)
            {
                /* set the recouce-argument, which will force a new allocation
                 *  even if the capacity would be large enough */
                if (capacity > count)
                    return reserve(count, count, single_ref, resource);
                return nullptr;
            }

            /*
             *  count: how many to at least require
             *  move: if realloc, how many to move-construct over (must be less than size)
             *  mr: if not zero, force-relloc to the new memory resource
             *      (even if its the same resource)
             *  returns null-container or new container if relocation occurred, in
             *  which case the new container will have a size of 'copy'
             */
            std::shared_ptr<self_type> reserve(size_type count, size_type move, bool single_ref,
                                               const mr::MemoryResource& mr)
            {
                if (count <= capacity && !mr)
                    return nullptr;
                mr::MemoryResource new_mr = bool(mr) ? mr : resource;

                /* check if this is a noninitial reserving for a growing size in which
                 *   case the capacity should not just satisfy the request but by some form
                 *   of equation to prevent to many reallocations */
                size_type new_cap = std::max<size_type>(count, move);
                if (new_cap > capacity && capacity > 0) {
                    size_type min_cap = new_cap;
                    new_cap = capacity;

                    if (capacity >= 4) {
                        while (new_cap < min_cap)
                            new_cap += (new_cap / 2);
                    } else {
                        while (new_cap < min_cap)
                            new_cap += new_cap;
                    }
                }

                /* check if an empty container has been requested */
                if (new_cap == 0)
                    return std::make_shared<self_type>(new_mr, nullptr, 0, 0);

                /* check if a relocation could suffice (not considered a container-change)
                 *  (can only occur, if this container has only one reference and is self-owned) */
                if (pointer != nullptr && new_mr == resource && !bool(release) && single_ref
                    && resource->tryResize(pointer, capacity * sizeof(value_type),
                                           alignof(value_type), new_cap * sizeof(value_type))) {
                    capacity = new_cap;
                    return nullptr;
                }

                /* perform the new allocation */
                raw_pointer new_ptr = static_cast<raw_pointer>(
                    new_mr->allocate(new_cap * sizeof(value_type), alignof(value_type)));
                std::shared_ptr<self_type> next =
                    std::make_shared<self_type>(new_mr, new_ptr, 0, new_cap);

                /* either move-initialize the data or perform the memory move */
                if constexpr (!is_trivial<TypeTag>) {
                    for (size_type i = 0; i < move; ++i) {
                        new (new_ptr + i) value_type(std::move(pointer[i]));
                        ++next->size;
                    }
                } else if (move > 0) {
                    copyMemory<value_type>(new_ptr, pointer, move, true);
                    next->size = move;
                }
                return next;
            }

            /*
             *  move [where; end] by diff within the container
             *  (where + diff) must be less or equal to size
             */
            void move_tail(size_type where, difference_type diff)
            {
                /* check if this move can be delegated to the mr */
                if constexpr (is_trivial<TypeTag>) {
                    moveMemory<value_type>(pointer + (where + diff), pointer + where,
                                           (size - where));
                    if (diff >= 0)
                        size = std::max<size_type>(size, (size - where) + diff);
                    else
                        size += diff;
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
                    new (pointer + size) value_type(std::move(pointer[next]));
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
            void insert_range(size_type off, ItType ibegin, size_type count, bool is_universal)
            {
                size_type end = off + count;

                /* check if the iterator has a 'base' function to get the raw pointer */
                if constexpr (detail::base_returns_pointer<ItType>::value
                              && detail::is_random_iterator<ItType>::value)
                    insert_range(off, ibegin.base(), count, is_universal);

                /* Check if the iterator has a 'get' function to get the raw pointer.
                 * Not checking for random access iterator, because we are checking for a
                 * thrust pointer type here, which is not tagged as random access iterator.
                 */
                else if constexpr (detail::get_returns_pointer<ItType>::value)
                    insert_range(off, ibegin.get(), count, is_universal);

                /* check if the iterator is a pointer of the right type
                 *   and can be reduced to a memory operation */
                else if constexpr (detail::actual_pointer<ItType>::value
                                   && std::is_same<std::decay_t<decltype(*ibegin)>,
                                                   std::decay_t<value_type>>::value
                                   && is_trivial<TypeTag>) {
                    copyMemory<value_type>(pointer + off, ibegin, count, is_universal);
                    size = std::max<size_type>(size, end);
                }

                /* insert the range by using the iterators */
                else {
                    static_cast<void>(is_universal);

                    size_type transition = std::min<size_type>(size, end);

                    /* copy-assign the currently existing and to-be-overwritten part */
                    while (off < transition) {
                        pointer[off] = *ibegin;
                        ++off;
                        ++ibegin;
                    }

                    /* copy-construct the remaining part */
                    while (size < end) {
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
                if (value == nullptr) {
                    /* check if the setting can be entirely skipped or delegated to the mr */
                    if constexpr (is_uninitialized<TypeTag>)
                        size = std::max<size_type>(size, end);
                    else if constexpr (is_trivial<TypeTag>) {
                        /* construct the one temporary value (only constructor is required,
                         *  destructor is not necessary nor expected from this type-tag) */
                        uint8_t tmp[sizeof(value_type)] = {0};
                        new (tmp) value_type();

                        fillMemory<value_type>(pointer + off, tmp, count);
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
                else if constexpr (is_trivial<TypeTag>) {
                    fillMemory<value_type>(pointer + off, value, count);
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
    } // namespace detail

    /*
     *  Return the state of the container and ensure its contents will
     *  not be released until the returned release-function has been called.
     */
    template <class Type>
    struct NativeContainer {
        std::function<void()> release;
        Type* raw_pointer = nullptr;
        size_t size = 0;
    };

    /*
     *  ContiguousVector behaves like an stl-vector (std::vector) with the difference of
     *      configuring its type behavior and allocations. Otherwise it uses copy/move
     *      semantics where possible/applicable.
     *
     *  Exceptions thrown by the type in the
     *          default-constructor
     *          move-constructor
     *          copy-constructor
     *          move-assignment operator
     *          copy-assignment operator
     *      or any iterators passed into this objects will leave the container
     *      in a valid state (i.e. proper cleanup is ensured) but a modified state.
     *    - For all other exceptions, atomic transactional behavior is guaranteed by the container.
     *
     *  PointerType and IteratorType must both be constructable from a raw Type-pointer.
     *
     *  In case of a trivial type_tag, the container will actively look for pointers as parameters
     *    to switch over to memory operations compared to iterator operations.
     *  It consideres the following iterators as pointers:
     *    - it is a pointer
     *    - it is a random access iterator, which has a 'base' member function, which returns a
     * pointer
     *    - it is a random access iterator, which has a 'get' member function, which returns a
     * pointer In all other scenarios, simple loops over iterators will occur, compared to the
     * potentially faster memory operations.
     *
     *  Each ContiguousVector is associated with a mr::MemoryResource (polymorphic allocators).
     *  All allocations and memory-operations are performed on the currently used resource.
     *  Changes to the current resource associated with this container:
     *    - copy-construct (optionally inherit the incoming resource)
     *    - move-construct (inherit the incoming resource)
     *    - other-constructors (parameter-resource or mr::defaultResource)
     *    - swap_resource calls
     *    - swap_content calls
     *    - swap() calls (optional parameter-resource else unchanged)
     *    - assign()/assign_default() calls (optional parameter-resource else unchanged)
     */
    template <class Type, class TypeTag, template <class> class PointerType,
              template <class> class IteratorType>
    class ContiguousVector
    {
    public:
        using self_type = ContiguousVector<Type, TypeTag, PointerType, IteratorType>;
        using container_type = detail::ContContainer<Type, TypeTag>;
        using value_type = typename container_type::value_type;
        using size_type = typename container_type::size_type;
        using difference_type = typename container_type::difference_type;

        using raw_pointer = typename container_type::raw_pointer;
        using const_raw_pointer = typename container_type::const_raw_pointer;
        using reference = typename container_type::reference;
        using const_reference = typename container_type::const_reference;

        using pointer = PointerType<Type>;
        using const_pointer = PointerType<const Type>;
        using iterator = IteratorType<Type>;
        using const_iterator = IteratorType<const Type>;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    private:
        template <class ItType>
        static constexpr bool is_universal_impl =
            std::is_same<ItType, iterator>::value || std::is_same<ItType, const_iterator>::value
            || std::is_same<ItType, pointer>::value || std::is_same<ItType, const_pointer>::value;
        template <class ItType>
        static constexpr bool is_universal = is_universal_impl<std::decay_t<ItType>>;

    private:
        std::shared_ptr<container_type> _self;

    private:
        /* reserves the given amount, updates the current container, and returns the old container
         */
        std::shared_ptr<container_type>
            _reserve(size_type count, size_type move,
                     const mr::MemoryResource& mr = mr::MemoryResource())
        {
            std::shared_ptr<container_type> _tmp = _self->reserve(count, move, _self.unique(), mr);

            /* check if a new container was allocated and swap them */
            if (_tmp == nullptr)
                return nullptr;
            std::swap(_self, _tmp);

            /* return the old container, if it is non-empty */
            if (_tmp->pointer == nullptr)
                return nullptr;
            return _tmp;
        }

        /* reduce the current container to the given amount */
        void _reduce(size_type count)
        {
            std::shared_ptr<container_type> _tmp = _self->reduce(count, _self.unique());
            if (_tmp == nullptr)
                return;
            _self = _tmp;
        }

    public:
        /* use defaultResource as initial resource */
        ContiguousVector()
        {
            _self = std::make_shared<container_type>(mr::defaultResource(), nullptr, 0, 0);
        }

        explicit ContiguousVector(size_type count)
        {
            _self = std::make_shared<container_type>(mr::defaultResource(), nullptr, 0, 0);

            _reserve(count, 0);
            _self->set_range(0, nullptr, count);
        }

        explicit ContiguousVector(size_type count, const_reference init)
        {
            _self = std::make_shared<container_type>(mr::defaultResource(), nullptr, 0, 0);

            _reserve(count, 0);
            _self->set_range(0, &init, count);
        }

        template <class ItType>
        ContiguousVector(ItType ibegin, ItType iend)
        {
            _self = std::make_shared<container_type>(mr::defaultResource(), nullptr, 0, 0);

            size_type count = std::distance(ibegin, iend);
            _reserve(count, 0);
            _self->insert_range(0, ibegin, count, is_universal<ItType>);
        }

        ContiguousVector(std::initializer_list<value_type> l)
        {
            _self = std::make_shared<container_type>(mr::defaultResource(), nullptr, 0, 0);

            _reserve(l.size(), 0);
            _self->insert_range(0, l.begin(), l.size(), false);
        }

        /* if mr is invalid, mr::defaultResource will be used */
        explicit ContiguousVector(const mr::MemoryResource& mr)
        {
            _self =
                std::make_shared<container_type>(!mr ? mr::defaultResource() : mr, nullptr, 0, 0);
        }

        explicit ContiguousVector(size_type count, const mr::MemoryResource& mr)
        {
            _self =
                std::make_shared<container_type>(!mr ? mr::defaultResource() : mr, nullptr, 0, 0);

            _reserve(count, 0);
            _self->set_range(0, nullptr, count);
        }

        explicit ContiguousVector(size_type count, const_reference init,
                                  const mr::MemoryResource& mr)
        {
            _self =
                std::make_shared<container_type>(!mr ? mr::defaultResource() : mr, nullptr, 0, 0);

            _reserve(count, 0);
            _self->set_range(0, &init, count);
        }

        template <class ItType>
        explicit ContiguousVector(ItType ibegin, ItType iend, const mr::MemoryResource& mr)
        {
            _self =
                std::make_shared<container_type>(!mr ? mr::defaultResource() : mr, nullptr, 0, 0);

            size_type count = std::distance(ibegin, iend);
            _reserve(count, 0);
            _self->insert_range(0, ibegin, count, is_universal<ItType>);
        }

        explicit ContiguousVector(std::initializer_list<value_type> l, const mr::MemoryResource& mr)
        {
            _self =
                std::make_shared<container_type>(!mr ? mr::defaultResource() : mr, nullptr, 0, 0);

            _reserve(l.size(), 0);
            _self->insert_range(0, l.begin(), l.size(), false);
        }

        /* use s::resource as initial resource */
        ContiguousVector(const self_type& s)
        {
            _self = std::make_shared<container_type>(s._self->resource, nullptr, 0, 0);

            _reserve(s._self->size, 0);
            _self->insert_range(0, s._self->pointer, s._self->size, true);
        }

        /* if mr is invalid, s::resource will be used */
        explicit ContiguousVector(const self_type& s, const mr::MemoryResource& mr)
        {
            _self = std::make_shared<container_type>(!mr ? s._self->resource : mr, nullptr, 0, 0);

            _reserve(s._self->size, 0);
            _self->insert_range(0, s._self->pointer, s._self->size, true);
        }

        /* use s::resource as initial resource */
        ContiguousVector(self_type&& s) noexcept
        {
            _self = std::make_shared<container_type>(s._self->resource, nullptr, 0, 0);
            std::swap(_self, s._self);
        }

        ~ContiguousVector() = default;

    public:
        /*
         *  Retrieve a copy of the underlying native container.
         *  The underlying memory will not be released until the release function has been invoked.
         *  Any relocations/releases of this ContiguousVector will allocate a new container, but
         *  leave the old container alive.
         */
        NativeContainer<Type> lock_native()
        {
            /* setup the release function to be a functor, which has a shared
             *  pointer and upon inocation it releases the shared pointer */
            struct CleanupFunctor {
                std::shared_ptr<container_type> _object;
                CleanupFunctor(const std::shared_ptr<container_type>& in) : _object(in) {}
                void operator()() { _object.reset(); }
            };

            NativeContainer<Type> _out;
            _out.raw_pointer = _self->pointer;
            _out.size = _self->size;
            _out.resize = CleanupFunctor(_self);
            return _out;
        }

        /*
         *  Setup the internal structure to point to the raw_pointer and use the size/capacity =
         *  size. The ContiguousVector will call destruct whenever the last reference to the
         *  resource has been released. The current resource will be stored and any
         *  relocation/release will allocate a new container with the old resource, thereby calling
         *  destruct on this object.
         */
        void from_extern(Type* raw_pointer, size_t size, std::function<void()> destruct)
        {
            _self = std::make_shared<container_type>(_self->resource, raw_pointer, size,
                                                     std::move(destruct));
        }

        /* if mr is invalid, mr::defaultResource will be used */
        mr::MemoryResource resource() const { return _self->resource; }

        void swap_resource(const mr::MemoryResource& mr)
        {
            mr::MemoryResource actual = bool(mr) ? mr : mr::defaultResource();
            if (actual == _self->resource)
                return;
            _reserve(0, _self->size, actual);
        }

        /* o::resourse will be used */
        void swap_content(self_type& o) { std::swap(_self, o._self); }

        /* current resource will be kept */
        self_type& operator=(const self_type& s)
        {
            if (&s == this)
                return *this;
            assign(s);
            return *this;
        }

        self_type& operator=(self_type&& s)
        {
            if (&s == this)
                return *this;
            if (_self->resource == s._self->resource)
                std::swap(_self, s._self);
            else
                assign(s);
            s._reduce(0);
            return *this;
        }

        self_type& operator=(std::initializer_list<value_type> l)
        {
            assign(l);
            return *this;
        }

        /* if mr is invalid, current resource will be used */
        void assign_default(size_type count, const mr::MemoryResource& mr = mr::MemoryResource())
        {
            _reserve(count, 0, mr);
            _self->set_range(0, nullptr, count);
            _self->destruct_until(count);
        }

        void assign(size_type count, const_reference init,
                    const mr::MemoryResource& mr = mr::MemoryResource())
        {
            _reserve(count, 0, mr);
            _self->set_range(0, &init, count);
            _self->destruct_until(count);
        }

        template <class ItType>
        void assign(ItType ibegin, ItType iend, const mr::MemoryResource& mr = mr::MemoryResource())
        {
            size_type count = std::distance(ibegin, iend);
            _reserve(count, 0, mr);
            _self->insert_range(0, ibegin, count, is_universal<ItType>);
            _self->destruct_until(count);
        }

        void assign(std::initializer_list<value_type> l,
                    const mr::MemoryResource& mr = mr::MemoryResource())
        {
            assign(l.begin(), l.end(), mr);
        }

        void assign(const self_type& s, const mr::MemoryResource& mr = mr::MemoryResource())
        {
            _reserve(s._self->size, 0, mr);
            _self->insert_range(0, s._self->pointer, s._self->size, true);
            _self->destruct_until(s._self->size);
        }

        /* if mr is invalid, current resource will be kept */
        void swap(self_type& o, const mr::MemoryResource& mr = mr::MemoryResource())
        {
            mr::MemoryResource actual = bool(mr) ? mr : _self->resource;

            if (_self->resource == actual && o._self->resource == actual)
                std::swap(_self, o._self);
            else {
                std::shared_ptr<container_type> _old = _reserve(o._self->size, 0, actual);

                _self->insert_range(0, o._self->pointer, o._self->size, true);

                o._reserve(_old == nullptr ? 0 : _old->size, 0);
                if (_old != nullptr)
                    o._self->insert_range(0, _old->pointer, _old->size, true);
                o._self->destruct_until(_old == nullptr ? 0 : _old->size);
            }
        }

        reference at(size_type i)
        {
            if (i >= _self->size)
                throw std::out_of_range("Index into ContiguousVector is out of range");
            return _self->pointer[i];
        }

        const_reference at(size_type i) const
        {
            if (i >= _self->size)
                throw std::out_of_range("Index into ContiguousVector is out of range");
            return _self->pointer[i];
        }

        reference operator[](size_type i) { return _self->pointer[i]; }

        const_reference operator[](size_type i) const { return _self->pointer[i]; }

        reference front() { return *_self->pointer; }

        const_reference front() const { return *_self->pointer; }

        reference back() { return *(_self->end_ptr() - 1); }

        const_reference back() const { return *(_self->end_ptr() - 1); }

        pointer data() { return pointer(_self->pointer); }

        const_pointer data() const { return const_pointer(_self->pointer); }

        iterator begin() { return iterator(_self->pointer); }

        const_iterator begin() const { return const_iterator(_self->pointer); }

        const_iterator cbegin() const { return const_iterator(_self->pointer); }

        iterator end() { return iterator(_self->end_ptr()); }

        const_iterator end() const { return const_iterator(_self->end_ptr()); }

        const_iterator cend() const { return const_iterator(_self->end_ptr()); }

        reverse_iterator rbegin() { return reverse_iterator(end()); }

        const_reverse_iterator rbegin() const { return const_reverse_iterator(cend()); }

        const_reverse_iterator crbegin() const { return const_reverse_iterator(cend()); }

        reverse_iterator rend() { return reverse_iterator(begin()); }

        const_reverse_iterator rend() const { return const_reverse_iterator(cbegin()); }

        const_reverse_iterator crend() const { return const_reverse_iterator(cbegin()); }

        size_type size() const { return _self->size; }

        bool empty() const { return size() == 0; }

        size_type max_size() const { return std::numeric_limits<difference_type>::max(); }

        void reserve(size_type cap) { _reserve(cap, _self->size); }

        size_type capacity() const { return _self->capacity; }

        void shrink_to_fit() { _reduce(_self->size); }

        void clear() noexcept { _self->destruct_until(0); }

        iterator insert_default(const_iterator i, size_type count)
        {
            difference_type off = i - const_iterator(_self->pointer);
            size_type pre = static_cast<size_type>(off);
            size_type post = _self->size - static_cast<size_type>(off);

            std::shared_ptr<container_type> _old = _reserve(_self->size + count, pre);

            /* check if a new buffer has been constructed */
            if (_old != nullptr) {
                _self->set_range(pre, nullptr, count);
                _self->insert_range(_self->size, _old->pointer + pre, post, true);
            }

            /* check if the current content will overlap itself upon moving */
            else if (count < post) {
                _self->move_tail(pre, count);
                _self->set_range(pre, nullptr, count);
            }

            else {
                /* construct the new part in order for the tail to be appended */
                _self->set_range(_self->size, nullptr, count - post);
                _self->insert_range(_self->size, _self->pointer + pre, post, true);
                _self->set_range(pre, nullptr, post);
            }
            return iterator(_self->pointer + off);
        }

        iterator insert(const_iterator i, const_reference value) { return emplace(i, value); }

        iterator insert(const_iterator i, value_type&& value)
        {
            return emplace(i, std::move(value));
        }

        iterator insert(const_iterator i, size_type count, const_reference value)
        {
            difference_type off = i - const_iterator(_self->pointer);
            size_type pre = static_cast<size_type>(off);
            size_type post = _self->size - static_cast<size_type>(off);

            std::shared_ptr<container_type> _old = _reserve(_self->size + count, pre);

            /* check if a new buffer has been constructed */
            if (_old != nullptr) {
                _self->set_range(pre, &value, count);
                _self->insert_range(_self->size, _old->pointer + pre, post, true);
            }

            /* check if the current content will overlap itself upon moving */
            else if (count < post) {
                _self->move_tail(pre, count);
                _self->set_range(pre, &value, count);
            }

            else {
                /* construct the new part in order for the tail to be appended */
                _self->set_range(_self->size, &value, count - post);
                _self->insert_range(_self->size, _self->pointer + pre, post, true);
                _self->set_range(pre, &value, post);
            }
            return iterator(_self->pointer + off);
        }

        template <class ItType>
        iterator insert(const_iterator i, ItType ibegin, ItType iend)
        {
            difference_type off = i - const_iterator(_self->pointer);
            size_type pre = static_cast<size_type>(off);
            size_type post = _self->size - static_cast<size_type>(off);
            size_type count = static_cast<size_type>(std::distance(ibegin, iend));

            std::shared_ptr<container_type> _old = _reserve(_self->size + count, pre);

            /* check if a new buffer has been constructed */
            if (_old != nullptr) {
                _self->insert_range(pre, ibegin, count, is_universal<ItType>);
                _self->insert_range(_self->size, _old->pointer + pre, post, true);
            }

            /* check if the current content will overlap itself upon moving */
            else if (count < post) {
                _self->move_tail(pre, count);
                _self->insert_range(pre, ibegin, count, is_universal<ItType>);
            }

            else {
                /* construct the new part in order for the tail to be appended */
                _self->insert_range(_self->size, std::next(ibegin, post), count - post,
                                    is_universal<ItType>);
                _self->insert_range(_self->size, _self->pointer + pre, post, true);
                _self->insert_range(pre, ibegin, post, is_universal<ItType>);
            }
            return iterator(_self->pointer + off);
        }

        iterator insert(const_iterator i, std::initializer_list<value_type> l)
        {
            return insert(i, l.begin(), l.end());
        }

        template <class... Args>
        iterator emplace(const_iterator i, Args&&... args)
        {
            size_type off = static_cast<size_type>(i - const_iterator(_self->pointer));
            std::shared_ptr<container_type> _old = _reserve(_self->size + 1, off);

            if (_old != nullptr) {
                new (_self->end_ptr()) value_type(std::forward<Args>(args)...);
                _self->insert_range(++_self->size, _old->pointer + off, _old->size - off, true);
            } else if (off >= _self->size) {
                new (_self->end_ptr()) value_type(std::forward<Args>(args)...);
                ++_self->size;
            } else {
                _self->move_tail(off, 1);
                _self->pointer[off] = std::move(value_type(std::forward<Args>(args)...));
            }
            return iterator(_self->pointer + off);
        }

        iterator erase(iterator i) { return erase(i, std::next(i)); }

        iterator erase(const_iterator i) { return erase(i, std::next(i)); }

        iterator erase(iterator ibegin, iterator iend)
        {
            return erase(const_iterator(ibegin), const_iterator(iend));
        }

        iterator erase(const_iterator ibegin, const_iterator iend)
        {
            difference_type offset = ibegin - const_iterator(_self->pointer);
            difference_type count = iend - ibegin;
            _self->move_tail(iend - const_iterator(_self->pointer), -count);
            return iterator(_self->pointer + offset);
        }

        template <class... Args>
        reference emplace_back(Args&&... args)
        {
            _reserve(_self->size + 1, _self->size);
            new (_self->end_ptr()) value_type(std::forward<Args>(args)...);
            return _self->pointer[_self->size++];
        }

        void push_back(const_reference value) { emplace_back(value); }

        void push_back(value_type&& value) { emplace_back(std::move(value)); }

        void pop_back() { _self->destruct_until(_self->size - 1); }

        void resize(size_type count)
        {
            if (count <= _self->size) {
                _self->destruct_until(count);
                return;
            }
            _reserve(count, _self->size);
            _self->set_range(_self->size, nullptr, count - _self->size);
        }

        void resize(size_type count, const_reference value)
        {
            if (count <= _self->size) {
                _self->destruct_until(count);
                return;
            }
            _reserve(count, _self->size);
            _self->set_range(_self->size, &value, count - _self->size);
        }
    };
} // namespace elsa::mr
