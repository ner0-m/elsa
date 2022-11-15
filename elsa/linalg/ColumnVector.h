#pragma once

#include "ContiguousStorage.h"

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>

namespace elsa::linalg
{
    // forward declare Vector class
    template <class data_t>
    class Vector;

    /**
     * @brief Mutable view of a column of a matrix. Somewhat of an implementation detail of the row
     * access of the `Matrix` class. The class behaves similar to the `Vector` class. Please note,
     * that no distinction is made from row to column vectors. You can convert a `RowView` to an
     * owning `Vector` with an constructor in the `Vector` class.
     *
     * The implementation idea was taken from:
     * - https://github.com/NVIDIA/thrust/blob/main/examples/strided_range.cu
     */
    template <class data_t>
    class ColumnView
    {
        using Iterator = typename ContiguousStorage<data_t>::iterator;

    public:
        using difference_type = typename thrust::iterator_difference<Iterator>::type;

        struct stride_functor {
            difference_type stride;

            stride_functor(difference_type stride) : stride(stride) {}

            __host__ __device__ difference_type operator()(const difference_type& i) const
            {
                return stride * i;
            }
        };

        // This is a bit of magic, but basicially it enables strided access to the iterator
        // This was taken from https://github.com/NVIDIA/thrust/blob/main/examples/strided_range.cu
        using CountingIterator = typename thrust::counting_iterator<difference_type>;
        using TransformIterator =
            typename thrust::transform_iterator<stride_functor, CountingIterator>;
        using PermutationIterator =
            typename thrust::permutation_iterator<Iterator, TransformIterator>;

        // type of the strided_range iterator
        using iterator = PermutationIterator;

        using size_type = std::ptrdiff_t;

        using value_type = data_t;
        using reference = value_type&;
        using const_reference = const value_type&;

        // construct strided_range for the range [first,last)
        ColumnView(Iterator first, Iterator last, difference_type stride)
            : first(first), last(last), stride(stride)
        {
        }

        iterator begin();

        iterator end();

        iterator begin() const;

        iterator end() const;

        iterator cbegin() const;

        iterator cend() const;

        size_type size() const;

        reference operator()(size_type idx);

        const_reference operator()(size_type idx) const;

        reference operator[](size_type idx);

        const_reference operator[](size_type idx) const;

        template <class T,
                  std::enable_if_t<std::is_convertible_v<T, data_t> && !std::is_same_v<T, data_t>,
                                   int> = 0>
        ColumnView& operator=(const T& val)
        {
            *this = static_cast<data_t>(val);
            return *this;
        }

        ColumnView& operator=(value_type val);

        ColumnView& operator=(const Vector<data_t>& v);

    protected:
        Iterator first;
        Iterator last;
        difference_type stride;
    };

    /**
     * @brief Mutable view of a column of a matrix. Somewhat of an implementation detail of the row
     * access of the `Matrix` class. The class behaves similar to the `Vector` class. Please note,
     * that no distinction is made from row to column vectors. You can convert a `RowView` to an
     * owning `Vector` with an constructor in the `Vector` class.
     *
     * The implementation idea was taken from:
     * - https://github.com/NVIDIA/thrust/blob/main/examples/strided_range.cu
     */
    template <class data_t>
    class ConstColumnView
    {
        using Iterator = typename ContiguousStorage<data_t>::const_iterator;

    public:
        using difference_type = typename thrust::iterator_difference<Iterator>::type;

        struct stride_functor {
            difference_type stride;

            stride_functor(difference_type stride) : stride(stride) {}

            __host__ __device__ difference_type operator()(const difference_type& i) const
            {
                return stride * i;
            }
        };

        // This is a bit of magic, but basicially it enables strided access to the iterator
        // This was taken from https://github.com/NVIDIA/thrust/blob/main/examples/strided_range.cu
        using CountingIterator = typename thrust::counting_iterator<difference_type>;
        using TransformIterator =
            typename thrust::transform_iterator<stride_functor, CountingIterator>;
        using PermutationIterator =
            typename thrust::permutation_iterator<Iterator, TransformIterator>;

        // type of the strided_range iterator
        using iterator = PermutationIterator;

        using size_type = std::ptrdiff_t;

        using value_type = data_t;
        using const_reference = const value_type&;

        // construct strided_range for the range [first,last)
        ConstColumnView(Iterator first, Iterator last, difference_type stride)
            : first(first), last(last), stride(stride)
        {
        }

        iterator begin() const;

        iterator end() const;

        iterator cbegin() const;

        iterator cend() const;

        size_type size() const;

        const_reference operator()(size_type idx) const;

        const_reference operator[](size_type idx) const;

    protected:
        Iterator first;
        Iterator last;
        difference_type stride;
    };
} // namespace elsa::linalg
