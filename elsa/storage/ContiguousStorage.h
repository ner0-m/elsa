#pragma once

#include "DisableWarnings.h"

DISABLE_WARNING_PUSH
DISABLE_WARNING_SIGN_CONVERSION
// Thrust is smart enough to always pick the correct vector for us
#include <thrust/universal_vector.h>
DISABLE_WARNING_POP

namespace elsa
{
    class Allocator
    {
    };

    Allocator* DefAllocator();

    template <class Type>
    class ContiguousPointer
    {
    public:
        using size_type = size_t;
        using difference_type = ptrdiff_t;
        using raw_pointer = Type*;

        using reference = Type&;

        Type& operator[](size_type i)
        {
            static Type _p;
            return _p;
        }
        const Type& operator[](size_type i) const
        {
            static Type _p;
            return _p;
        }

        ContiguousPointer<Type> operator+(difference_type r) const
        {
            return ContiguousPointer<Type>();
        }

        Type* get() { return 0; }
        const Type* get() const { return 0; }
    };

    template <class Type>
    class ContiguousStorage_
    {
    public:
        using iterator = Type*;
        using const_iterator = const Type*;
        using reference = Type&;
        using const_reference = const Type&;

        using pointer = ContiguousPointer<Type>;
        using const_pointer = const pointer;

        using size_type = typename pointer::size_type;
        using difference_type = typename pointer::difference_type;

    private:
        Allocator* _Alloc = 0;

    public:
        ContiguousStorage_(Allocator* a = DefAllocator()) {}
        ContiguousStorage_(size_type sz) {}

        template <class ItType>
        ContiguousStorage_(ItType b, ItType e)
        {
        }

        ContiguousStorage_(std::initializer_list<Type> l) {}

        template <class ItType>
        void assign(ItType a, ItType b)
        {
        }

        iterator begin() { return 0; }
        const_iterator begin() const { return 0; }
        const_iterator cbegin() const { return 0; }
        iterator end() { return 0; }
        const_iterator end() const { return 0; }
        const_iterator cend() const { return 0; }

        size_type size() const { return 0; }
        bool empty() const { return size() == 0; }

        pointer data() { return pointer(); }
        const_pointer data() const { return pointer(); }

        reference operator[](size_type i)
        {
            static Type _p;
            return _p;
        }
        const_reference operator[](size_type i) const
        {
            static Type _p;
            return _p;
        }
    };

    template <class T>
    using ContiguousStorage = thrust::universal_vector<T>;
} // namespace elsa
