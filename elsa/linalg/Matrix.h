#pragma once

#include "ContiguousStorage.h"
#include "RowVector.h"
#include "ColumnVector.h"
#include "Vector.h"

#include <Eigen/Core>
#include <initializer_list>
#include <cstddef>

namespace elsa::linalg
{
    /**
     * @brief Represent a dynamic mathematical n x m matrix. The matrix is stored always stored in
     * row major order. The interface somewhat resembles the Eigen interface to matrices, but it's
     * way less powerful and complete!
     *
     * @note If you need proper linear algebra support, use an already existing framework!
     */
    template <class data_t>
    class Matrix
    {
    public:
        using size_type = std::ptrdiff_t;

        using value_type = data_t;
        using reference = value_type&;
        using const_reference = const value_type&;

        using pointer = value_type*;
        using const_pointer = const value_type*;

        using iterator = typename ContiguousStorage<data_t>::iterator;
        using const_iterator = typename ContiguousStorage<data_t>::const_iterator;

        Matrix() = default;

        /// Defaulted copy constructor
        Matrix(const Matrix&);

        /// Defaulted copy assignment
        Matrix& operator=(const Matrix&);

        /// Defaulted move constructor
        Matrix(Matrix&&) noexcept;

        /// Defaulted move assignment
        Matrix& operator=(Matrix&&) noexcept;

        /// Construct matrix of size m x n
        Matrix(size_type rows, size_type cols);

        /// Construct matrix of size m x n with given value
        Matrix(size_type rows, size_type cols, const_reference val);

        /// Construct matrix of size m x n with given initialize list
        Matrix(size_type rows, size_type cols, std::initializer_list<data_t> list);

        /// Construct from dynamic Eigen Matrix with row major storage order
        explicit Matrix(
            const Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& mat);

        /// Construct from dynamic Eigen Matrix with column major storage order
        explicit Matrix(
            const Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& mat);

        /// Fill vector with scalar of non `data_t` type to vector
        template <class T,
                  std::enable_if_t<std::is_convertible_v<T, data_t> && !std::is_same_v<T, data_t>,
                                   int> = 0>
        Matrix& operator=(const T& val)
        {
            *this = static_cast<data_t>(val);
            return *this;
        }

        /// Fill vector with scalar value
        Matrix& operator=(value_type val);

        /// return pointer to beginning of storage
        pointer data();

        /// return canst pointer to beginning of storage
        const_pointer data() const;

        /// return begin iterator
        iterator begin();

        /// return end iterator
        iterator end();

        /// return begin const_iterator
        const_iterator begin() const;

        /// return end const_iterator
        const_iterator end() const;

        /// return begin const_iterator
        const_iterator cbegin() const;

        /// return end const_iterator
        const_iterator cend() const;

        /// return the number of rows of the matrix
        size_type rows() const;

        /// return the number of cols of the matrix
        size_type cols() const;

        /// return a mutable reference to the `(i, j)` element
        reference operator()(size_type i, size_type j);

        /// return a const reference to the `(i, j)` element
        const_reference operator()(size_type i, size_type j) const;

        /// return a mutable view of the `i`-th column into the matrix
        ColumnView<data_t> col(size_type i);

        /// return a const view of the `i`-th column into the matrix
        ConstColumnView<data_t> col(size_type i) const;

        /// return a mutable view of the `i`-th row into the matrix
        RowView<data_t> row(size_type i);

        /// return a const view of the `i`-th row into the matrix
        ConstRowView<data_t> row(size_type i) const;

        /// Reshape, only if newrows * newcols is equal to the previous size. The values are kept
        /// and reinterpreted according to the new sizes
        void reshape(size_type newrows, size_type newcols);

        /// Resize to given size. This will allocate a new buffer, and it is unspecified to access
        /// any values of the matrix until new values are assigned.
        void resize(size_type newrows, size_type newcols);

        /// return the transpose of the matrix
        Matrix transpose() const;

    private:
        /// little helper to row-major indexing
        size_type index(size_type i, size_type j) const { return i * cols_ + j; }

        size_type rows_;
        size_type cols_;
        ContiguousStorage<data_t> storage_;
    };

    /// Matrix-Vector product. Reasonable fast, but does not dispatch to a BLAS implementation, so
    /// especially for large matrices and vectors this will be slower, but then again, this is not a
    /// linear algebra library.
    ///
    /// @tparam T is either `Vector`, `RowView` or `ColumnView`
    template <class data_t, class T>
    Vector<data_t> operator*(const Matrix<data_t>& mat, const T& x);

    template <class data_t>
    std::ostream& operator<<(std::ostream& stream, const Matrix<data_t>& mat);
} // namespace elsa::linalg
