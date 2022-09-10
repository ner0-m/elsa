#pragma once

#include "elsaDefines.h"
#include "DataDescriptor.h"
#include "DataHandler.h"
#include "DataHandlerCPU.h"
#include "DataHandlerMapCPU.h"
#include "DataContainerIterator.h"
#include "Error.h"
#include "FormatConfig.h"
#include "TypeCasts.hpp"

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
     * - David Frank - added DataHandler concept, iterators
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

        /// delete default constructor (without metadata there can be no valid container)
        DataContainer() = delete;

        /**
         * @brief Constructor for empty DataContainer, no initialisation is performed,
         *        but the underlying space is allocated.
         *
         * @param[in] dataDescriptor containing the associated metadata
         * @param[in] handlerType the data handler (default: CPU)
         */
        explicit DataContainer(const DataDescriptor& dataDescriptor,
                               DataHandlerType handlerType = defaultHandlerType);

        /**
         * @brief Constructor for DataContainer, initializing it with a DataVector
         *
         * @param[in] dataDescriptor containing the associated metadata
         * @param[in] data vector containing the initialization data
         * @param[in] handlerType the data handler (default: CPU)
         */
        DataContainer(const DataDescriptor& dataDescriptor,
                      const Eigen::Matrix<data_t, Eigen::Dynamic, 1>& data,
                      DataHandlerType handlerType = defaultHandlerType);

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
        DataContainer<data_t>& operator=(DataContainer<data_t>&& other);

        /// return the current DataDescriptor
        const DataDescriptor& getDataDescriptor() const;

        /// return the size of the stored data (i.e. the number of elements in the linearized
        /// signal)
        index_t getSize() const;

        /// return the index-th element of linearized signal (not bounds-checked!)
        data_t& operator[](index_t index);

        /// return the index-th element of the linearized signal as read-only (not bounds-checked!)
        const data_t& operator[](index_t index) const;

        /// return an element by n-dimensional coordinate (not bounds-checked!)
        data_t& operator()(const IndexVector_t& coordinate);

        /// return an element by n-dimensional coordinate as read-only (not bounds-checked!)
        const data_t& operator()(const IndexVector_t& coordinate) const;

        data_t at(const IndexVector_t& coordinate) const;

        /// return an element by its coordinates (not bounds-checked!)
        template <typename idx0_t, typename... idx_t,
                  typename = std::enable_if_t<
                      std::is_integral_v<idx0_t> && (... && std::is_integral_v<idx_t>)>>
        data_t& operator()(idx0_t idx0, idx_t... indices)
        {
            IndexVector_t coordinate(sizeof...(indices) + 1);
            ((coordinate << idx0), ..., indices);
            return operator()(coordinate);
        }

        /// return an element by its coordinates as read-only (not bounds-checked!)
        template <typename idx0_t, typename... idx_t,
                  typename = std::enable_if_t<
                      std::is_integral_v<idx0_t> && (... && std::is_integral_v<idx_t>)>>
        const data_t& operator()(idx0_t idx0, idx_t... indices) const
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
        void fft(FFTNorm norm) const;

        /// convert to the inverse fourier transformed signal
        void ifft(FFTNorm norm) const;

        /// if the datacontainer is already complex, return itself.
        template <typename _data_t = data_t>
        typename std::enable_if_t<isComplex<_data_t>, DataContainer<_data_t>> asComplex() const
        {
            return *this;
        }

        /// if the datacontainer is not complex,
        /// return a copy and fill in 0 as imaginary values
        template <typename _data_t = data_t>
        typename std::enable_if_t<not isComplex<_data_t>, DataContainer<complex<_data_t>>>
            asComplex() const
        {
            DataContainer<complex<data_t>> ret{
                *this->_dataDescriptor,
                this->_dataHandlerType,
            };

            // extend with complex zero value
            for (index_t idx = 0; idx < this->getSize(); ++idx) {
                ret[idx] = complex<data_t>{(*this)[idx], 0};
            }

            return ret;
        }

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

        /// iterator for DataContainer (random access and continuous)
        using iterator = DataContainerIterator<DataContainer<data_t>>;

        /// const iterator for DataContainer (random access and continuous)
        using const_iterator = ConstDataContainerIterator<DataContainer<data_t>>;

        /// alias for reverse iterator
        using reverse_iterator = std::reverse_iterator<iterator>;
        /// alias for const reverse iterator
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;

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

        /// returns reversed iterator to the last element of the container
        reverse_iterator rbegin();

        /// returns const reversed iterator to the last element of the container (cannot mutate
        /// data)
        const_reverse_iterator rbegin() const;

        /// returns const reversed iterator to the last element of the container (cannot mutate
        /// data)
        const_reverse_iterator crbegin() const;

        /// returns reversed iterator to one past the first element of container
        reverse_iterator rend();

        /// returns const reversed iterator to one past the first element of container (cannot
        /// mutate data)
        const_reverse_iterator rend() const;

        /// returns const reversed iterator to one past the first element of container (cannot
        /// mutate data)
        const_reverse_iterator crend() const;

        /// value_type of the DataContainer elements for iterators
        using value_type = data_t;
        /// pointer type of DataContainer elements for iterators
        using pointer = data_t*;
        /// const pointer type of DataContainer elements for iterators
        using const_pointer = const data_t*;
        /// reference type of DataContainer elements for iterators
        using reference = data_t&;
        /// const reference type of DataContainer elements for iterators
        using const_reference = const data_t&;
        /// difference type for iterators
        using difference_type = std::ptrdiff_t;

        /// returns the type of the DataHandler in use
        DataHandlerType getDataHandlerType() const;

        /// friend constexpr function to implement expression templates
        template <bool GPU, class Operand, std::enable_if_t<isDataContainer<Operand>, int>>
        friend constexpr auto evaluateOrReturn(Operand const& operand);

        /// write a pretty-formatted string representation to stream
        void format(std::ostream& os, format_config cfg = {}) const;

        /**
         * @brief Factory function which returns GPU based DataContainers
         *
         * @return the GPU based DataContainer
         *
         * Note that if this function is called on a container which is already GPU based, it
         * will throw an exception.
         */
        DataContainer loadToGPU();

        /**
         * @brief Factory function which returns CPU based DataContainers
         *
         * @return the CPU based DataContainer
         *
         * Note that if this function is called on a container which is already CPU based, it will
         * throw an exception.
         */
        DataContainer loadToCPU();

    private:
        /// the current DataDescriptor
        std::unique_ptr<DataDescriptor> _dataDescriptor;

        /// the current DataHandler
        std::unique_ptr<DataHandler<data_t>> _dataHandler;

        /// the current DataHandlerType
        DataHandlerType _dataHandlerType;

        /// factory method to create DataHandlers based on handlerType with perfect forwarding of
        /// constructor arguments
        template <typename... Args>
        std::unique_ptr<DataHandler<data_t>> createDataHandler(DataHandlerType handlerType,
                                                               Args&&... args);

        /// private constructor accepting a DataDescriptor and a DataHandler
        explicit DataContainer(const DataDescriptor& dataDescriptor,
                               std::unique_ptr<DataHandler<data_t>> dataHandler,
                               DataHandlerType dataType = defaultHandlerType);

        /**
         * @brief Helper function to indicate if a regular assignment or a clone should be performed
         *
         * @param[in] handlerType the member variable of the other container in
         * copy-/move-assignment
         *
         * @return true if a regular assignment of the pointed to DataHandlers should be done
         *
         * An assignment operation with a DataContainer which does not use the same device (CPU /
         * GPU) has to be handled differently. This helper function indicates if a regular
         * assignment should be performed or not.
         */
        bool canAssign(DataHandlerType handlerType);
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
    DataContainer<data_t> clip(DataContainer<data_t> dc, data_t min, data_t max);

    /// Concatenate two DataContainers to one (requires copying of both)
    template <typename data_t>
    DataContainer<data_t> concatenate(const DataContainer<data_t>& dc1,
                                      const DataContainer<data_t>& dc2);

    /// Perform the FFT shift operation to the provided signal. Refer to
    /// https://numpy.org/doc/stable/reference/generated/numpy.fft.fftshift.html for further
    /// details.
    template <typename data_t>
    DataContainer<data_t> fftShift2D(DataContainer<data_t> dc);

    /// Perform the IFFT shift operation to the provided signal. Refer to
    /// https://numpy.org/doc/stable/reference/generated/numpy.fft.ifftshift.html for further
    /// details.
    template <typename data_t>
    DataContainer<data_t> ifftShift2D(DataContainer<data_t> dc);

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
              typename = std::enable_if_t<std::is_arithmetic_v<GetFloatingPointType_t<Scalar>>>>
    inline DataContainer<data_t> operator*(const DataContainer<data_t>& dc, const Scalar& s)
    {
        auto copy = dc;
        copy *= s;
        return copy;
    }

    template <typename data_t, typename Scalar,
              typename = std::enable_if_t<std::is_arithmetic_v<GetFloatingPointType_t<Scalar>>>>
    inline DataContainer<data_t> operator*(const Scalar& s, const DataContainer<data_t>& dc)
    {
        auto copy = dc;
        copy *= s;
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
              typename = std::enable_if_t<std::is_arithmetic_v<GetFloatingPointType_t<Scalar>>>>
    inline DataContainer<data_t> operator+(const DataContainer<data_t>& dc, const Scalar& s)
    {
        auto copy = dc;
        copy += s;
        return copy;
    }

    template <typename data_t, typename Scalar,
              typename = std::enable_if_t<std::is_arithmetic_v<GetFloatingPointType_t<Scalar>>>>
    inline DataContainer<data_t> operator+(const Scalar& s, const DataContainer<data_t>& dc)
    {
        auto copy = dc;
        copy += s;
        return copy;
    }

    /// Subtract two DataContainers
    template <typename data_t>
    inline DataContainer<data_t> operator-(const DataContainer<data_t>& lhs,
                                           const DataContainer<data_t>& rhs)
    {
        auto copy = lhs;
        copy -= rhs;
        return copy;
    }

    template <typename data_t, typename Scalar,
              typename = std::enable_if_t<std::is_arithmetic_v<GetFloatingPointType_t<Scalar>>>>
    inline DataContainer<data_t> operator-(const DataContainer<data_t>& dc, const Scalar& s)
    {
        auto copy = dc;
        copy -= s;
        return copy;
    }

    template <typename data_t, typename Scalar,
              typename = std::enable_if_t<std::is_arithmetic_v<GetFloatingPointType_t<Scalar>>>>
    inline DataContainer<data_t> operator-(const Scalar& s, const DataContainer<data_t>& dc)
    {
        auto copy = dc;
        std::transform(copy.begin(), copy.end(), copy.begin(), [=](auto val) { return s - val; });
        return copy;
    }

    /// Divide two DataContainers
    template <typename data_t>
    inline DataContainer<data_t> operator/(const DataContainer<data_t>& lhs,
                                           const DataContainer<data_t>& rhs)
    {
        auto copy = lhs;
        copy /= rhs;
        return copy;
    }

    template <typename data_t, typename Scalar,
              typename = std::enable_if_t<std::is_arithmetic_v<GetFloatingPointType_t<Scalar>>>>
    inline DataContainer<data_t> operator/(const DataContainer<data_t>& dc, const Scalar& s)
    {
        auto copy = dc;
        copy /= s;
        return copy;
    }

    template <typename data_t, typename Scalar,
              typename = std::enable_if_t<std::is_arithmetic_v<GetFloatingPointType_t<Scalar>>>>
    inline DataContainer<data_t> operator/(const Scalar& s, const DataContainer<data_t>& dc)
    {
        auto copy = dc;
        std::transform(copy.begin(), copy.end(), copy.begin(), [=](auto val) { return s / val; });
        return copy;
    }

    template <typename data_t>
    inline DataContainer<data_t> cwiseMax(const DataContainer<std::complex<data_t>>& lhs,
                                          const DataContainer<std::complex<data_t>>& rhs)
    {
        DataContainer<data_t> copy(rhs.getDataDescriptor());
        std::transform(lhs.begin(), lhs.end(), rhs.begin(), copy.begin(),
                       [](auto l, auto r) { return std::max(std::abs(l), std::abs(r)); });
        return copy;
    }

    template <typename data_t>
    inline DataContainer<data_t> cwiseMax(const DataContainer<std::complex<data_t>>& lhs,
                                          const DataContainer<data_t>& rhs)
    {
        DataContainer<data_t> copy(rhs.getDataDescriptor());
        std::transform(lhs.begin(), lhs.end(), rhs.begin(), copy.begin(),
                       [](auto l, auto r) { return std::max(std::abs(l), r); });
        return copy;
    }

    template <typename data_t>
    inline DataContainer<data_t> cwiseMax(const DataContainer<data_t>& lhs,
                                          const DataContainer<std::complex<data_t>>& rhs)
    {
        DataContainer<data_t> copy(rhs.getDataDescriptor());
        std::transform(lhs.begin(), lhs.end(), rhs.begin(), copy.begin(),
                       [](auto l, auto r) { return std::max(l, std::abs(r)); });
        return copy;
    }

    template <typename data_t, typename = std::enable_if_t<!isComplex<data_t>>>
    inline DataContainer<data_t> cwiseMax(const DataContainer<data_t>& lhs,
                                          const DataContainer<data_t>& rhs)
    {
        DataContainer<data_t> copy(rhs.getDataDescriptor());
        std::transform(lhs.begin(), lhs.end(), rhs.begin(), copy.begin(),
                       [](auto l, auto r) { return std::max(l, r); });
        return copy;
    }

    template <typename data_t>
    inline DataContainer<data_t> cwiseAbs(const DataContainer<data_t>& dc)
    {
        DataContainer<data_t> copy(dc.getDataDescriptor());
        std::transform(dc.begin(), dc.end(), copy.begin(), [](auto l) { return std::abs(l); });
        return copy;
    }

    template <typename data_t>
    inline DataContainer<data_t> square(const DataContainer<data_t>& dc)
    {
        DataContainer<data_t> copy(dc.getDataDescriptor());
        std::transform(dc.begin(), dc.end(), copy.begin(), [](auto x) { return x * x; });
        return copy;
    }

    template <typename data_t>
    inline DataContainer<data_t> sqrt(const DataContainer<data_t>& dc)
    {
        DataContainer<data_t> copy(dc.getDataDescriptor());
        std::transform(dc.begin(), dc.end(), copy.begin(), [](auto x) { return std::sqrt(x); });
        return copy;
    }

    template <typename data_t>
    inline DataContainer<data_t> exp(const DataContainer<data_t>& dc)
    {
        DataContainer<data_t> copy(dc.getDataDescriptor());
        std::transform(dc.begin(), dc.end(), copy.begin(), [](auto x) { return std::exp(x); });
        return copy;
    }

    template <typename data_t>
    inline DataContainer<data_t> log(const DataContainer<data_t>& dc)
    {
        DataContainer<data_t> copy(dc.getDataDescriptor());
        std::transform(dc.begin(), dc.end(), copy.begin(), [](auto x) { return std::log(x); });
        return copy;
    }

    /// Real for complex DataContainers
    template <typename data_t>
    inline DataContainer<data_t> real(const DataContainer<std::complex<data_t>>& dc)
    {
        DataContainer<data_t> result(dc.getDataDescriptor());
        std::transform(dc.begin(), dc.end(), result.begin(), [](auto x) { return std::real(x); });
        return result;
    }

    /// Real for real DataContainers, just a copy
    template <typename data_t, typename = std::enable_if_t<!isComplex<data_t>>>
    inline DataContainer<data_t> real(const DataContainer<data_t>& dc)
    {
        return dc;
    }

    /// Imag for complex DataContainers
    template <typename data_t>
    inline DataContainer<data_t> imag(const DataContainer<std::complex<data_t>>& dc)
    {
        DataContainer<data_t> result(dc.getDataDescriptor());
        std::transform(dc.begin(), dc.end(), result.begin(), [](auto x) { return std::imag(x); });
        return result;
    }

    /// Imag for real DataContainers, returns zero
    template <typename data_t, typename = std::enable_if_t<!isComplex<data_t>>>
    inline DataContainer<data_t> imag(const DataContainer<data_t>& dc)
    {
        DataContainer<data_t> result(dc.getDataDescriptor());
        result = 0;
        return result;
    }
} // namespace elsa
