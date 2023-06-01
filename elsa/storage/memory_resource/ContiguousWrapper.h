#pragma once

#include <cinttypes>
#include <type_traits>

/* define guards for __host__ and __device__ tags used by cuda */
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

namespace elsa::mr::detail
{
    /*
     *  Wraps a pointer and offers 'get' function to retrieve it.
     *  Pointer is expected to be accessible on hosts and devices.
     */
    template <class Type>
    class ContPointer
    {
    public:
        using self_type = ContPointer<Type>;
        using value_type = std::remove_cv_t<Type>;
        using size_type = size_t;
        using difference_type = std::ptrdiff_t;

        using raw_pointer = Type*;
        using pointer = Type*;
        using reference = Type&;

    private:
        pointer _where = 0;

    public:
        __host__ __device__ ContPointer() {}
        __host__ __device__ ContPointer(pointer w) : _where(w) {}
        __host__ __device__ ContPointer(self_type&& p) noexcept : _where(p._where) {}
        __host__ __device__ ContPointer(const ContPointer<std::remove_const_t<Type>>& p) : _where(p.get()) {}
        __host__ __device__ ContPointer(const ContPointer<std::add_const_t<Type>>& p) : _where(p.get()) {
            static_assert(std::is_const<Type>::value, "Const pointer cannot be converted to a normal pointer");
        }

    public:
        __host__ __device__ raw_pointer get() const { return _where; }

    public:
        __host__ __device__ self_type& operator=(const self_type& p)
        {
            _where = p._where;
            return *this;
        }
        __host__ __device__ self_type& operator=(self_type&& p) noexcept
        {
            _where = p._where;
            return *this;
        }
        __host__ __device__ bool operator==(const self_type& p) const { return _where == p._where; }
        __host__ __device__ bool operator!=(const self_type& p) const { return !(*this == p); }
        __host__ __device__ reference operator*() const { return *_where; }
        __host__ __device__ pointer operator->() const { return _where; }
        __host__ __device__ self_type& operator++()
        {
            ++_where;
            return *this;
        };
        __host__ __device__ self_type operator++(int)
        {
            self_type out(_where);
            ++_where;
            return out;
        }
        __host__ __device__ self_type& operator--()
        {
            --_where;
            return *this;
        };
        __host__ __device__ self_type operator--(int)
        {
            self_type out(_where);
            --_where;
            return out;
        }

        __host__ __device__ self_type& operator+=(difference_type d)
        {
            _where += d;
            return *this;
        }
        __host__ __device__ self_type& operator-=(difference_type d)
        {
            _where -= d;
            return *this;
        }
        __host__ __device__ self_type operator+(difference_type d) const
        {
            return self_type(_where + d);
        }
        __host__ __device__ self_type operator-(difference_type d) const
        {
            return self_type(_where - d);
        }
        __host__ __device__ difference_type operator-(const self_type& p) const
        {
            return _where - p._where;
        }
        __host__ __device__ reference operator[](size_type i) const { return _where[i]; }
        __host__ __device__ bool operator<(const self_type& p) const { return _where < p._where; }
        __host__ __device__ bool operator<=(const self_type& p) const { return _where <= p._where; }
        __host__ __device__ bool operator>=(const self_type& p) const { return !(*this < p); }
        __host__ __device__ bool operator>(const self_type& p) const { return !(*this <= p); }
    };

    /*
     *  Wraps a pointer to a contiguous (random-access) range of values.
     *  Pointer is expected to be accessible on hosts and devices.
     */
    template <class Type>
    class ContIterator
    {
    public:
        using self_type = ContIterator<Type>;
        using value_type = std::remove_cv_t<Type>;
        using iterator_category = std::random_access_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using size_type = size_t;
        using pointer = Type*;
        using reference = Type&;

    private:
        pointer _where = 0;

    public:
        __host__ __device__ ContIterator() {}
        __host__ __device__ ContIterator(pointer w) : _where(w) {}
        __host__ __device__ ContIterator(self_type&& p) noexcept : _where(p._where) {}
        __host__ __device__ ContIterator(const ContIterator<std::remove_const_t<Type>>& p) : _where(p.base()) {}
        __host__ __device__ ContIterator(const ContIterator<std::add_const_t<Type>>& p) : _where(p.base()) {
            static_assert(std::is_const<Type>::value, "Const iterator cannot be converted to a normal iterator");
        }

    public:
        __host__ __device__ self_type& operator=(const self_type& p)
        {
            _where = p._where;
            return *this;
        }
        __host__ __device__ self_type& operator=(self_type&& p) noexcept
        {
            _where = p._where;
            return *this;
        }
        __host__ __device__ bool operator==(const self_type& p) const { return _where == p._where; }
        __host__ __device__ bool operator!=(const self_type& p) const { return !(*this == p); }
        __host__ __device__ reference operator*() const { return *_where; }
        __host__ __device__ pointer operator->() const { return _where; }
        __host__ __device__ self_type& operator++()
        {
            ++_where;
            return *this;
        };
        __host__ __device__ self_type operator++(int)
        {
            self_type out(_where);
            ++_where;
            return out;
        }
        __host__ __device__ self_type& operator--()
        {
            --_where;
            return *this;
        };
        __host__ __device__ self_type operator--(int)
        {
            self_type out(_where);
            --_where;
            return out;
        }

        __host__ __device__ self_type& operator+=(difference_type d)
        {
            _where += d;
            return *this;
        }
        __host__ __device__ self_type& operator-=(difference_type d)
        {
            _where -= d;
            return *this;
        }
        __host__ __device__ self_type operator+(difference_type d) const
        {
            return self_type(_where + d);
        }
        __host__ __device__ self_type operator-(difference_type d) const
        {
            return self_type(_where - d);
        }
        __host__ __device__ difference_type operator-(const self_type& p) const
        {
            return _where - p._where;
        }
        __host__ __device__ reference operator[](size_type i) const { return _where[i]; }
        __host__ __device__ bool operator<(const self_type& p) const { return _where < p._where; }
        __host__ __device__ bool operator<=(const self_type& p) const { return _where <= p._where; }
        __host__ __device__ bool operator>=(const self_type& p) const { return !(*this < p); }
        __host__ __device__ bool operator>(const self_type& p) const { return !(*this <= p); }
        __host__ __device__ pointer base() const { return _where; }
    };
} // namespace elsa::mr::detail
