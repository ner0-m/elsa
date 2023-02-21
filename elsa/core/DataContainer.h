#pragma once

#include "TypeTraits.hpp"
#include "elsaDefines.h"
#include "ExpressionPredicates.h"
#include "DataDescriptor.h"
#include "Error.h"
#include "FormatConfig.h"
#include "TypeCasts.hpp"
#include "ContiguousStorage.h"
#include "Span.h"
#include "Overloaded.hpp"

#include <thrust/complex.h>

#include <variant>
#include <memory>
#include <type_traits>

namespace elsa
{
    /**
     * @brief class representing and storing a linearized n-dimensional signal
     *
     * This class provides a container for a signal that is stored in memory. This signal can
     * be n-dimensional, and will be stored in memory in a linearized fashion. The information
     * on how this linearization is performed is provided by an associated DataDescriptor.
     *
     * @tparam data_t data type that is stored in the DataContainer, defaulting to real_t.
     *
     * @author
     * - Matthias Wieczorek - initial code
     * - Tobias Lasser - rewrite, modularization, modernization
     * - David Frank - added DataHandler concept, iterators, integrated unified memory
     * - Nikola Dinev - add block support
     * - Jens Petit - expression templates
     * - Jonas Jelten - various enhancements, fft, complex handling, pretty formatting
     */
    template <typename data_t>
    class DataContainer
    {
    public:
        /// Scalar alias
        using Scalar = data_t;

        using reference = typename ContiguousStorage<data_t>::reference;
        using const_reference = typename ContiguousStorage<data_t>::const_reference;

        /// iterator for DataContainer (random access and continuous)
        using iterator = typename ContiguousStorage<data_t>::iterator;

        /// const iterator for DataContainer (random access and continuous)
        using const_iterator = typename ContiguousStorage<data_t>::const_iterator;

        /// delete default constructor (without metadata there can be no valid container)
        DataContainer() = delete;

        /**
         * @brief Constructor for empty DataContainer, no initialisation is performed,
         *        but the underlying space is allocated.
         *
         * @param[in] dataDescriptor containing the associated metadata
         */
        explicit DataContainer(const DataDescriptor& dataDescriptor);

        /**
         * @brief Constructor for DataContainer, initializing it with a DataVector
         *
         * @param[in] dataDescriptor containing the associated metadata
         * @param[in] data vector containing the initialization data
         */
        DataContainer(const DataDescriptor& dataDescriptor,
                      const Eigen::Matrix<data_t, Eigen::Dynamic, 1>& data);

        /// constructor accepting a DataDescriptor and a DataHandler
        DataContainer(const DataDescriptor& dataDescriptor,
                      const ContiguousStorage<data_t>& storage);

        /// constructor accepting a DataDescriptor and a DataHandler
        DataContainer(const DataDescriptor& dataDescriptor, ContiguousStorageView<data_t> storage);

        /**
         * @brief Copy constructor for DataContainer
         *
         * @param[in] other DataContainer to copy
         */
        DataContainer(const DataContainer<data_t>& other);

        /**
         * @brief copy assignment for DataContainer
         *
         * @param[in] other DataContainer to copy
         *
         * Note that a copy assignment with a DataContainer on a different device (CPU vs GPU) will
         * result in an "infectious" copy which means that afterwards the current container will use
         * the same device as "other".
         */
        DataContainer<data_t>& operator=(const DataContainer<data_t>& other);

        /**
         * @brief Move constructor for DataContainer
         *
         * @param[in] other DataContainer to move from
         *
         * The moved-from objects remains in a valid state. However, as preconditions are not
         * fulfilled for any member functions, the object should not be used. After move- or copy-
         * assignment, this is possible again.
         */
        DataContainer(DataContainer<data_t>&& other) noexcept;

        /**
         * @brief Move assignment for DataContainer
         *
         * @param[in] other DataContainer to move from
         *
         * The moved-from objects remains in a valid state. However, as preconditions are not
         * fulfilled for any member functions, the object should not be used. After move- or copy-
         * assignment, this is possible again.
         *
         * Note that a copy assignment with a DataContainer on a different device (CPU vs GPU) will
         * result in an "infectious" copy which means that afterwards the current container will use
         * the same device as "other".
         */
        DataContainer<data_t>& operator=(DataContainer<data_t>&& other) noexcept;

        /// return the current DataDescriptor
        const DataDescriptor& getDataDescriptor() const;

        /// return true, if the current DataContainer is owning its memory
        bool isOwning() const;

        /// return true, if the current DataContainer is a view, i.e. is not owning its memory
        bool isView() const;

        ContiguousStorage<data_t>& storage();

        const ContiguousStorage<data_t>& storage() const;

        /// return the size of the stored data (i.e. the number of elements in the linearized
        /// signal)
        index_t getSize() const;

        /// return the index-th element of linearized signal (not bounds-checked!)
        reference operator[](index_t index);

        /// return the index-th element of the linearized signal as read-only (not bounds-checked!)
        const_reference operator[](index_t index) const;

        /// return an element by n-dimensional coordinate (not bounds-checked!)
        reference operator()(const IndexVector_t& coordinate);

        /// return an element by n-dimensional coordinate as read-only (not bounds-checked!)
        const_reference operator()(const IndexVector_t& coordinate) const;

        data_t at(const IndexVector_t& coordinate) const;

        /// return an element by its coordinates (not bounds-checked!)
        template <typename idx0_t, typename... idx_t,
                  typename = std::enable_if_t<
                      std::is_integral_v<idx0_t> && (... && std::is_integral_v<idx_t>)>>
        reference operator()(idx0_t idx0, idx_t... indices)
        {
            IndexVector_t coordinate(sizeof...(indices) + 1);
            ((coordinate << idx0), ..., indices);
            return operator()(coordinate);
        }

        /// return an element by its coordinates as read-only (not bounds-checked!)
        template <typename idx0_t, typename... idx_t,
                  typename = std::enable_if_t<
                      std::is_integral_v<idx0_t> && (... && std::is_integral_v<idx_t>)>>
        const_reference operator()(idx0_t idx0, idx_t... indices) const
        {
            IndexVector_t coordinate(sizeof...(indices) + 1);
            ((coordinate << idx0), ..., indices);
            return operator()(coordinate);
        }

        /// return the dot product of this signal with the one from container other
        data_t dot(const DataContainer<data_t>& other) const;

        /// return the squared l2 norm of this signal (dot product with itself)
        GetFloatingPointType_t<data_t> squaredL2Norm() const;

        /// return the l2 norm of this signal (square root of dot product with itself)
        GetFloatingPointType_t<data_t> l2Norm() const;

        /// return the l0 pseudo-norm of this signal (number of non-zero values)
        index_t l0PseudoNorm() const;

        /// return the l1 norm of this signal (sum of absolute values)
        GetFloatingPointType_t<data_t> l1Norm() const;

        /// return the linf norm of this signal (maximum of absolute values)
        GetFloatingPointType_t<data_t> lInfNorm() const;

        /// return the sum of all elements of this signal
        data_t sum() const;

        /// return the min of all elements of this signal
        data_t minElement() const;

        /// return the max of all elements of this signal
        data_t maxElement() const;

        /// convert to the fourier transformed signal
        void fft(FFTNorm norm);

        /// convert to the inverse fourier transformed signal
        void ifft(FFTNorm norm);

        /// if the datacontainer is already complex, return itself.
        DataContainer<add_complex_t<data_t>> asComplex() const;

        /// compute in-place element-wise addition of another container
        DataContainer<data_t>& operator+=(const DataContainer<data_t>& dc);

        /// compute in-place element-wise subtraction of another container
        DataContainer<data_t>& operator-=(const DataContainer<data_t>& dc);

        /// compute in-place element-wise multiplication with another container
        DataContainer<data_t>& operator*=(const DataContainer<data_t>& dc);

        /// compute in-place element-wise division by another container
        DataContainer<data_t>& operator/=(const DataContainer<data_t>& dc);

        /// compute in-place addition of a scalar
        DataContainer<data_t>& operator+=(data_t scalar);

        /// compute in-place subtraction of a scalar
        DataContainer<data_t>& operator-=(data_t scalar);

        /// compute in-place multiplication with a scalar
        DataContainer<data_t>& operator*=(data_t scalar);

        /// compute in-place division by a scalar
        DataContainer<data_t>& operator/=(data_t scalar);

        /// assign a scalar to the DataContainer
        DataContainer<data_t>& operator=(data_t scalar);

        /// comparison with another DataContainer
        bool operator==(const DataContainer<data_t>& other) const;

        /// comparison with another DataContainer
        bool operator!=(const DataContainer<data_t>& other) const;

        /// returns a reference to the i-th block, wrapped in a DataContainer
        DataContainer<data_t> getBlock(index_t i);

        /// returns a const reference to the i-th block, wrapped in a DataContainer
        const DataContainer<data_t> getBlock(index_t i) const;

        /// return a view of this DataContainer with a different descriptor
        DataContainer<data_t> viewAs(const DataDescriptor& dataDescriptor);

        /// return a const view of this DataContainer with a different descriptor
        const DataContainer<data_t> viewAs(const DataDescriptor& dataDescriptor) const;

        /// @brief Slice the container in the last dimension
        ///
        /// Access a portion of the container via a slice. The slicing always is in the last
        /// dimension. So for a 3D volume, the slice would be an sliced in the z direction and would
        /// be a part of the x-y plane.
        ///
        /// A slice is always the same dimension as the original DataContainer, but with a thickness
        /// of 1 in the last dimension (i.e. the coefficient of the last dimension is 1)
        const DataContainer<data_t> slice(index_t i) const;

        /// @brief Slice the container in the last dimension, non-const overload
        ///
        /// @overload
        /// @see slice(index_t) const
        DataContainer<data_t> slice(index_t i);

        /// returns iterator to the first element of the container
        iterator begin();

        /// returns const iterator to the first element of the container (cannot mutate data)
        const_iterator begin() const;

        /// returns const iterator to the first element of the container (cannot mutate data)
        const_iterator cbegin() const;

        /// returns iterator to one past the last element of the container
        iterator end();

        /// returns const iterator to one past the last element of the container (cannot mutate
        /// data)
        const_iterator end() const;

        /// returns const iterator to one past the last element of the container (cannot mutate
        /// data)
        const_iterator cend() const;

        /// value_type of the DataContainer elements for iterators
        using value_type = data_t;
        /// pointer type of DataContainer elements for iterators
        using pointer = data_t*;
        /// const pointer type of DataContainer elements for iterators
        using const_pointer = const data_t*;
        /// difference type for iterators
        using difference_type = std::ptrdiff_t;

        /// write a pretty-formatted string representation to stream
        void format(std::ostream& os, format_config cfg = {}) const;

    private:
        /// the current DataDescriptor
        std::unique_ptr<DataDescriptor> _dataDescriptor;

        /// the current DataHandler
        std::variant<ContiguousStorage<data_t>, ContiguousStorageView<data_t>> storage_;
    };

    /// pretty output formatting.
    /// for configurable output, use `DataContainerFormatter` directly.
    template <typename T>
    std::ostream& operator<<(std::ostream& os, const elsa::DataContainer<T>& dc)
    {
        dc.format(os);
        return os;
    }

    /// clip the container values outside of the interval, to the interval edges
    template <typename data_t>
    DataContainer<data_t> clip(const DataContainer<data_t>& dc, data_t min, data_t max);

    /// Concatenate two DataContainers to one (requires copying of both)
    template <typename data_t>
    DataContainer<data_t> concatenate(const DataContainer<data_t>& dc1,
                                      const DataContainer<data_t>& dc2);

    /// Perform the FFT shift operation to the provided signal. Refer to
    /// https://numpy.org/doc/stable/reference/generated/numpy.fft.fftshift.html for further
    /// details.
    template <typename data_t>
    DataContainer<data_t> fftShift2D(const DataContainer<data_t>& dc);

    /// Perform the IFFT shift operation to the provided signal. Refer to
    /// https://numpy.org/doc/stable/reference/generated/numpy.fft.ifftshift.html for further
    /// details.
    template <typename data_t>
    DataContainer<data_t> ifftShift2D(const DataContainer<data_t>& dc);

    /// Unary plus operator
    template <typename data_t>
    inline DataContainer<data_t> operator+(const DataContainer<data_t>& x)
    {
        return x;
    }

    /// Unary negation operator
    template <typename data_t>
    inline DataContainer<data_t> operator-(const DataContainer<data_t>& x)
    {
        return static_cast<data_t>(-1) * x;
    }

    /// Multiplying two DataContainers
    template <typename data_t>
    inline DataContainer<data_t> operator*(const DataContainer<data_t>& lhs,
                                           const DataContainer<data_t>& rhs)
    {
        auto copy = lhs;
        copy *= rhs;
        return copy;
    }

    template <typename data_t, typename Scalar,
              typename = std::enable_if_t<std::is_arithmetic_v<GetFloatingPointType_t<
                                              Scalar>> && std::is_convertible_v<Scalar, data_t>>>
    inline DataContainer<data_t> operator*(const DataContainer<data_t>& dc, const Scalar& s)
    {
        auto copy = dc;
        copy *= static_cast<data_t>(s);
        return copy;
    }

    template <typename data_t, typename Scalar,
              typename = std::enable_if_t<std::is_arithmetic_v<GetFloatingPointType_t<
                                              Scalar>> && std::is_convertible_v<Scalar, data_t>>>
    inline DataContainer<data_t> operator*(const Scalar& s, const DataContainer<data_t>& dc)
    {
        auto copy = dc;
        copy *= static_cast<data_t>(s);
        return copy;
    }

    /// Add two DataContainers
    template <typename data_t>
    inline DataContainer<data_t> operator+(const DataContainer<data_t>& lhs,
                                           const DataContainer<data_t>& rhs)
    {
        auto copy = lhs;
        copy += rhs;
        return copy;
    }

    template <typename data_t, typename Scalar,
              typename = std::enable_if_t<std::is_arithmetic_v<GetFloatingPointType_t<
                                              Scalar>> && std::is_convertible_v<Scalar, data_t>>>
    inline DataContainer<data_t> operator+(const DataContainer<data_t>& dc, const Scalar& s)
    {
        auto copy = dc;
        copy += static_cast<data_t>(s);
        return copy;
    }

    template <typename data_t, typename Scalar,
              typename = std::enable_if_t<std::is_arithmetic_v<GetFloatingPointType_t<
                                              Scalar>> && std::is_convertible_v<Scalar, data_t>>>
    inline DataContainer<data_t> operator+(const Scalar& s, const DataContainer<data_t>& dc)
    {
        auto copy = dc;
        copy += static_cast<data_t>(s);
        return copy;
    }

    /// Subtract two DataContainers
    template <typename data_t>
    DataContainer<data_t> operator-(const DataContainer<data_t>& lhs,
                                    const DataContainer<data_t>& rhs);

    template <typename data_t, typename Scalar,
              typename = std::enable_if_t<std::is_arithmetic_v<GetFloatingPointType_t<
                                              Scalar>> && std::is_convertible_v<Scalar, data_t>>>
    DataContainer<std::common_type_t<data_t, Scalar>> operator-(const DataContainer<data_t>& dc,
                                                                const Scalar& s);

    template <typename Scalar, typename data_t,
              typename = std::enable_if_t<std::is_arithmetic_v<GetFloatingPointType_t<
                                              Scalar>> && std::is_convertible_v<Scalar, data_t>>>
    DataContainer<std::common_type_t<Scalar, data_t>> operator-(const Scalar& s,
                                                                const DataContainer<data_t>& dc);

    /// Divide two DataContainers
    template <typename data_t>
    DataContainer<data_t> operator/(const DataContainer<data_t>& lhs,
                                    const DataContainer<data_t>& rhs);

    /// Divide DataContainer by scalar
    template <typename data_t, typename Scalar,
              typename = std::enable_if_t<std::is_arithmetic_v<GetFloatingPointType_t<
                                              Scalar>> && std::is_convertible_v<Scalar, data_t>>>
    DataContainer<std::common_type_t<data_t, Scalar>> operator/(const DataContainer<data_t>& dc,
                                                                const Scalar& s);

    /// Divide scalar with DataContainer
    template <typename Scalar, typename data_t,
              typename = std::enable_if_t<std::is_arithmetic_v<GetFloatingPointType_t<
                                              Scalar>> && std::is_convertible_v<Scalar, data_t>>>
    DataContainer<std::common_type_t<Scalar, data_t>> operator/(const Scalar& s,
                                                                const DataContainer<data_t>& dc);

    template <typename xdata_t, typename ydata_t>
    DataContainer<value_type_of_t<std::common_type_t<xdata_t, ydata_t>>>
        cwiseMax(const DataContainer<xdata_t>& lhs, const DataContainer<ydata_t>& rhs);

    template <typename xdata_t, typename ydata_t>
    DataContainer<value_type_of_t<std::common_type_t<xdata_t, ydata_t>>>
        cwiseMin(const DataContainer<xdata_t>& lhs, const DataContainer<ydata_t>& rhs);

    /// @brief Compute a coefficient wise square for each element of the `DataContainer`
    template <typename data_t>
    DataContainer<data_t> square(const DataContainer<data_t>& dc);

    /// @brief Compute a coefficient wise square root for each element of the `DataContainer`
    template <typename data_t>
    DataContainer<data_t> sqrt(const DataContainer<data_t>& dc);

    /// @brief Compute a coefficient wise exponential for each element of the `DataContainer`
    template <typename data_t>
    DataContainer<data_t> exp(const DataContainer<data_t>& dc);

    /// @brief Compute a coefficient wise log for each element of the `DataContainer`
    template <typename data_t>
    DataContainer<data_t> log(const DataContainer<data_t>& dc);

    /// @brief Compute a coefficient wise minimum with a scalar.
    /// For each element in `x_i` the given `DataContainer`, compute
    /// `min(x_i, scalar)`
    template <typename data_t>
    DataContainer<data_t> minimum(const DataContainer<data_t>& dc, SelfType_t<data_t> scalar);

    /// @brief Compute a coefficient wise maximum with a scalar.
    /// For each element in `x_i` the given `DataContainer`, compute
    /// `max(x_i, scalar)`
    template <typename data_t>
    DataContainer<data_t> maximum(const DataContainer<data_t>& dc, SelfType_t<data_t> scalar);

    /// Return an owning DataContainer, if given an non-owning one, the data is copied to a new
    /// owning buffer.
    template <class data_t>
    DataContainer<data_t> materialize(const DataContainer<data_t>& x);

    template <typename data_t>
    DataContainer<value_type_of_t<data_t>> cwiseAbs(const DataContainer<data_t>& dc);

    template <typename data_t>
    DataContainer<add_complex_t<data_t>> asComplex(const DataContainer<data_t>& dc);

    /// Real for complex DataContainers
    template <typename data_t>
    DataContainer<value_type_of_t<data_t>> real(const DataContainer<data_t>& dc);

    /// Imag for complex DataContainers
    template <typename data_t>
    DataContainer<value_type_of_t<data_t>> imag(const DataContainer<data_t>& dc);

    /// Compute the linear combination of \f$a * x + b * y\f$.
    ///
    /// This function can be used as a memory efficient version for the computation
    /// of the above expression, as for such an expression (without expression template)
    /// multiple copies need to be created and allocated.
    ///
    /// The function throws, if x and y do not have the same data descriptor
    template <class data_t>
    DataContainer<data_t> lincomb(SelfType_t<data_t> a, const DataContainer<data_t>& x,
                                  SelfType_t<data_t> b, const DataContainer<data_t>& y);

    /// Compute the linear combination of \f$a * x + b * y\f$, and write it to
    /// the output variable.
    ///
    /// This function can be used as a memory efficient version for the computation
    /// of the above expression, as for such an expression (without expression template)
    /// multiple copies need to be created and allocated.
    ///
    /// The function throws, if x, y and out do not have the same data descriptor
    template <class data_t>
    void lincomb(SelfType_t<data_t> alpha, const DataContainer<data_t>& x, SelfType_t<data_t> b,
                 const DataContainer<data_t>& y, DataContainer<data_t>& out);

} // namespace elsa
