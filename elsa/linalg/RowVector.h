#pragma once

#include "ContiguousStorage.h"

#include "thrust/iterator/iterator_traits.h"

namespace elsa::linalg
{
    // forward declare Vector class
    template <class data_t>
    class Vector;

    /**
     * @brief Mutable view of a row of a matrix. Somewhat of an implementation detail of the row
     * access of the `Matrix` class. The class behaves similar to the `Vector` class. Please note,
     * that no distinction is made from row to column vectors. You can convert a `RowView` to an
     * owning `Vector` with an constructor in the `Vector` class.
     */
    template <class data_t>
    class RowView
    {
    public:
        using iterator = typename ContiguousStorage<data_t>::iterator;

        using size_type = std::ptrdiff_t;
        using difference_type = typename thrust::iterator_difference<iterator>::type;

        using value_type = data_t;
        using reference = value_type&;
        using const_reference = const value_type&;

        RowView(iterator first, iterator last);

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
        RowView& operator=(const T& val)
        {
            *this = static_cast<data_t>(val);
            return *this;
        }

        RowView& operator=(value_type val);

        RowView& operator=(const Vector<data_t>& v);

    protected:
        iterator first;
        iterator last;
    };

    /**
     * @brief Const view of a row of a matrix. Somewhat of an implementation detail of the row
     * access of the `Matrix` class. The class behaves similar to the `Vector` class. Please note,
     * that no distinction is made from row to column vectors. You can convert a `RowView` to an
     * owning `Vector` with an constructor in the `Vector` class.
     */
    template <class data_t>
    class ConstRowView
    {
    public:
        using iterator = typename ContiguousStorage<data_t>::const_iterator;

        using size_type = std::ptrdiff_t;
        using difference_type = typename thrust::iterator_difference<iterator>::type;

        using value_type = data_t;
        using const_reference = const value_type&;

        ConstRowView(iterator first, iterator last);

        iterator begin() const;

        iterator end() const;

        iterator cbegin() const;

        iterator cend() const;

        size_type size() const;

        const_reference operator()(size_type idx) const;

        const_reference operator[](size_type idx) const;

    protected:
        iterator first;
        iterator last;
    };
} // namespace elsa::linalg
