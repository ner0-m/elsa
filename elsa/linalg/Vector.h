#pragma once

#include "ContiguousStorage.h"
#include "RowVector.h"
#include "ColumnVector.h"

#include <Eigen/Core>

#include <cstddef>
#include <initializer_list>

namespace elsa::linalg
{
    /**
     * @brief Represent a dynamic mathematical vector using unified memory. From an interface it
     * follows somewhat the Eigen interface for vectors. Of course, this class is only a fraction as
     * powerful as Eigen's or any other linear algebras vector class. This is mostly a wrapper
     * around the unified memory.
     *
     * @note If you need proper linear algebra support, use an already existing framework!
     */
    template <class data_t>
    class Vector
    {
    public:
        using size_type = std::ptrdiff_t;

        using value_type = data_t;
        using reference = value_type&;
        using const_reference = const value_type&;

        using iterator = typename ContiguousStorage<data_t>::iterator;
        using const_iterator = typename ContiguousStorage<data_t>::const_iterator;

        Vector() = default;
        Vector(const Vector&) = default;
        Vector& operator=(const Vector&) = default;
        Vector(Vector&&) = default;
        Vector& operator=(Vector&&) = default;

        /// Construct vector of given size
        explicit Vector(size_type size);

        /// Construct vector of given size with value
        Vector(size_type size, const_reference val);

        /// Construct vector with initialize list
        explicit Vector(std::initializer_list<data_t> list);

        /// Construct vector with ContiguousStorage
        explicit Vector(const ContiguousStorage<data_t>& storage);

        /// Construct vector from iterator pair
        Vector(iterator first, iterator end);

        /// Construct from row vector
        explicit Vector(RowView<data_t> row);

        /// Construct from const row vector
        explicit Vector(ConstRowView<data_t> row);

        /// Construct from column vector
        explicit Vector(ColumnView<data_t> col);

        /// Construct from const column vector
        explicit Vector(ConstColumnView<data_t> col);

        /// Construct from Eigen Matrix
        explicit Vector(Eigen::Matrix<data_t, Eigen::Dynamic, 1> mat);

        /// Return the undyling storage
        ContiguousStorage<data_t>& storage() { return storage_; }

        /// Return the undyling storage
        const ContiguousStorage<data_t>& storage() const { return storage_; }

        /// Return the raw underlying pointer
        data_t* data() { return thrust::raw_pointer_cast(storage_.data()); }

        /// Return the raw underlying pointer
        const data_t* data() const { return thrust::raw_pointer_cast(storage_.data()); }

        /// Return an iterator to the begin of the memory region
        iterator begin();

        /// Return an iterator one past the end of the memory region
        iterator end();

        /// Return an const iterator to the begin of the memory region
        const_iterator begin() const;

        /// Return an const iterator one past the end of the memory region
        const_iterator end() const;

        /// Return an const iterator to the begin of the memory region
        const_iterator cbegin() const;

        /// Return an const iterator one past the end of the memory region
        const_iterator cend() const;

        /// Return the size of the vector
        size_type size() const;

        /// Return a reference to the `idx`-th element
        reference operator()(size_type idx);

        /// Return a const reference to the `idx`-th element
        const_reference operator()(size_type idx) const;

        /// Return a reference to the `idx`-th element
        reference operator[](size_type idx);

        /// Return a const reference to the `idx`-th element
        const_reference operator[](size_type idx) const;

        /// Assign a scalar to the vector. The scalar type must be convertible to the underlying
        /// storage type.
        template <class T,
                  std::enable_if_t<std::is_convertible_v<T, data_t> && !std::is_same_v<T, data_t>,
                                   int> = 0>
        Vector& operator=(const T& val)
        {
            *this = static_cast<data_t>(val);
            return *this;
        }

        /// Assign a scalar to the vector
        Vector& operator=(value_type val);

        /// In-place coefficient wise addition with scalar
        Vector& operator+=(data_t s);

        /// In-place coefficient wise subtraction with scalar
        Vector& operator-=(data_t s);

        /// In-place coefficient wise multiplication with scalar
        Vector& operator*=(data_t s);

        /// In-place coefficient wise division with scalar
        Vector& operator/=(data_t s);

        /// In-place coefficient wise addition with other vector
        Vector& operator+=(const Vector<data_t>& v);

        /// In-place coefficient wise subtraction with other vector
        Vector& operator-=(const Vector<data_t>& v);

        /// In-place coefficient wise multiplication with other vector
        Vector& operator*=(const Vector<data_t>& v);

        /// In-place coefficient wise division with other vector
        Vector& operator/=(const Vector<data_t>& v);

    private:
        ContiguousStorage<data_t> storage_;
    };

    /// User defined deduction guide
    template <class data_t>
    explicit Vector(Eigen::Matrix<data_t, Eigen::Dynamic, 1> mat) -> Vector<data_t>;

    /// Vector scalar addition
    template <class data_t>
    Vector<data_t> operator+(const Vector<data_t>& v, data_t s);

    template <
        class data_t, class T,
        std::enable_if_t<std::is_convertible_v<T, data_t> && !std::is_same_v<T, data_t>, int> = 0>
    Vector<data_t> operator+(const Vector<data_t>& v, T s)
    {
        return v + static_cast<data_t>(s);
    }

    template <class T, class data_t, std::enable_if_t<std::is_convertible_v<T, data_t>, int> = 0>
    Vector<data_t> operator+(T s, const Vector<data_t>& v)
    {
        return v + s;
    }

    /// Vector scalar subtraction
    template <class data_t>
    Vector<data_t> operator-(const Vector<data_t>& v, data_t s);

    template <
        class data_t, class T,
        std::enable_if_t<std::is_convertible_v<T, data_t> && !std::is_same_v<T, data_t>, int> = 0>
    Vector<data_t> operator-(const Vector<data_t>& v, T s)
    {
        return v - static_cast<data_t>(s);
    }

    template <class T, class data_t, std::enable_if_t<std::is_convertible_v<T, data_t>, int> = 0>
    Vector<data_t> operator-(T s, const Vector<data_t>& v)
    {
        return (-s) + v;
    }

    /// Vector scalar multiplication
    template <class data_t>
    Vector<data_t> operator*(const Vector<data_t>& v, data_t s);

    template <
        class data_t, class T,
        std::enable_if_t<std::is_convertible_v<T, data_t> && !std::is_same_v<T, data_t>, int> = 0>
    Vector<data_t> operator*(const Vector<data_t>& v, T s)
    {
        return v * static_cast<data_t>(s);
    }

    template <class T, class data_t, std::enable_if_t<std::is_convertible_v<T, data_t>, int> = 0>
    Vector<data_t> operator*(T s, const Vector<data_t>& v)
    {
        return v * static_cast<data_t>(s);
    }

    /// Vector scalar division
    template <class data_t>
    Vector<data_t> operator/(const Vector<data_t>& v, data_t s);

    template <
        class data_t, class T,
        std::enable_if_t<std::is_convertible_v<T, data_t> && !std::is_same_v<T, data_t>, int> = 0>
    Vector<data_t> operator/(const Vector<data_t>& v, T s)
    {
        return v / static_cast<data_t>(s);
    }

    /// Coefficient wise vector addition
    template <class data_t>
    Vector<data_t> operator+(const Vector<data_t>& x, const Vector<data_t>& y);

    /// Coefficient wise vector subtraction
    template <class data_t>
    Vector<data_t> operator-(const Vector<data_t>& x, const Vector<data_t>& y);

    /// Coefficient wise vector multiplication
    template <class data_t>
    Vector<data_t> operator*(const Vector<data_t>& x, const Vector<data_t>& y);

    /// Coefficient wise vector division
    template <class data_t>
    Vector<data_t> operator/(const Vector<data_t>& x, const Vector<data_t>& y);

    /// Normalize the given vector \f$ v = v / | v | \f$
    template <class data_t>
    void normalize(Vector<data_t>& v);

    /// Return a normalized copy of the given vector \f$ x = v / | v | \f$
    template <class data_t>
    Vector<data_t> normalized(const Vector<data_t>& v);

    /// Compute the \f$\ell^2\f$ norm of the given vector
    template <class data_t>
    data_t norm(const Vector<data_t>& v);

    /// Compute the scalar product of the two vectors
    template <class data_t>
    data_t dot(const Vector<data_t>& x, const Vector<data_t>& y);

    template <class data_t>
    std::ostream& operator<<(std::ostream& stream, const Vector<data_t>& v);
} // namespace elsa::linalg
