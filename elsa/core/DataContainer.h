#pragma once

#include "elsaDefines.h"
#include "DataDescriptor.h"
#include "DataHandler.h"
#include "DataHandlerCPU.h"
#include "DataHandlerMapCPU.h"
#include "DataContainerIterator.h"
#include "Error.h"
#include "Expression.h"
#include "TypeCasts.hpp"
#include <limits>

#ifdef ELSA_CUDA_VECTOR
#include "DataHandlerGPU.h"
#include "DataHandlerMapGPU.h"
#endif

#include <iostream>
#include <memory>
#include <type_traits>

namespace elsa
{

    /**
     * @brief class representing and storing a linearized n-dimensional signal
     *
     * @author Matthias Wieczorek - initial code
     * @author Tobias Lasser - rewrite, modularization, modernization
     * @author David Frank - added DataHandler concept, iterators
     * @author Nikola Dinev - add block support
     * @author Jens Petit - expression templates
     * @author Jonas Jelten - various enhancements, fft, complex handling, pretty formatting
     *
     * @tparam data_t - data type that is stored in the DataContainer, defaulting to real_t.
     *
     * This class provides a container for a signal that is stored in memory. This signal can
     * be n-dimensional, and will be stored in memory in a linearized fashion. The information
     * on how this linearization is performed is provided by an associated DataDescriptor.
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

        /**
         * @brief Expression evaluation assignment for DataContainer
         *
         * @param[in] source expression to evaluate
         *
         * This evaluates an expression template term into the underlying data member of
         * the DataHandler in use.
         */
        template <typename Source, typename = std::enable_if_t<isExpression<Source>>>
        DataContainer<data_t>& operator=(Source const& source)
        {
            if (auto handler = downcast_safe<DataHandlerCPU<data_t>>(_dataHandler.get())) {
                handler->accessData() = source.template eval<false>();
            } else if (auto handler =
                           downcast_safe<DataHandlerMapCPU<data_t>>(_dataHandler.get())) {
                handler->accessData() = source.template eval<false>();
#ifdef ELSA_CUDA_VECTOR
            } else if (auto handler = downcast_safe<DataHandlerGPU<data_t>>(_dataHandler.get())) {
                handler->accessData().eval(source.template eval<true>());
            } else if (auto handler =
                           downcast_safe<DataHandlerMapGPU<data_t>>(_dataHandler.get())) {
                handler->accessData().eval(source.template eval<true>());
#endif
            } else {
                throw LogicError("Unknown handler type");
            }

            return *this;
        }

        /**
         * @brief Expression constructor
         *
         * @param[in] source expression to evaluate
         *
         * It creates a new DataContainer out of an expression. For this the meta information which
         * is saved in the expression is used.
         */
        template <typename Source, typename = std::enable_if_t<isExpression<Source>>>
        DataContainer<data_t>(Source const& source)
            : DataContainer<data_t>(source.getDataMetaInfo().first, source.getDataMetaInfo().second)
        {
            this->operator=(source);
        }

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
        data_t& operator()(IndexVector_t coordinate);

        /// return an element by n-dimensional coordinate as read-only (not bounds-checked!)
        const data_t& operator()(IndexVector_t coordinate) const;

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

        /// return the dot product of this signal with the one from an expression
        template <typename Source, typename = std::enable_if_t<isExpression<Source>>>
        data_t dot(const Source& source) const
        {
            if (auto handler = downcast_safe<DataHandlerCPU<data_t>>(_dataHandler.get())) {
                return (*this * source).template eval<false>().sum();
            } else if (auto handler =
                           downcast_safe<DataHandlerMapCPU<data_t>>(_dataHandler.get())) {
                return (*this * source).template eval<false>().sum();
#ifdef ELSA_CUDA_VECTOR
            } else if (auto handler = downcast_safe<DataHandlerGPU<data_t>>(_dataHandler.get())) {
                DataContainer temp = (*this * source);
                return temp.sum();
            } else if (auto handler =
                           downcast_safe<DataHandlerMapGPU<data_t>>(_dataHandler.get())) {
                DataContainer temp = (*this * source);
                return temp.sum();
#endif
            } else {
                throw LogicError("Unknown handler type");
            }
        }

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

        /// convert to the fourier transformed signal
        void fft() const;

        /// convert to the inverse fourier transformed signal
        void ifft() const;

        /// if the datacontainer is already complex, return itself.
        template <typename _data_t = data_t>
        typename std::enable_if_t<isComplex<_data_t>, DataContainer<_data_t>> asComplex() const
        {
            return *this;
        }

        /// if the datacontainer is not complex,
        /// return a copy and fill in 0 as imaginary values
        template <typename _data_t = data_t>
        typename std::enable_if_t<not isComplex<_data_t>, DataContainer<std::complex<_data_t>>>
            asComplex() const
        {
            DataContainer<std::complex<data_t>> ret{
                *this->_dataDescriptor,
                this->_dataHandlerType,
            };

            // extend with complex zero value
            for (index_t idx = 0; idx < this->getSize(); ++idx) {
                ret[idx] = std::complex<data_t>{(*this)[idx], 0};
            }

            return ret;
        }

        /// get only the real part and discard the imaginary values
        template <typename _data_t = data_t>
        typename std::enable_if_t<isComplex<_data_t>,
                                  DataContainer<GetFloatingPointType_t<_data_t>>>
            getReal() const
        {
            return this->getComplexSplitup<true>();
        }

        /// get only the imaginary part and discard the real values
        template <typename _data_t = data_t>
        typename std::enable_if_t<isComplex<_data_t>,
                                  DataContainer<GetFloatingPointType_t<_data_t>>>
            getImaginary() const
        {
            return this->getComplexSplitup<false>();
        }

        /// return the minimum element of all the stored values
        template <typename _data_t = data_t>
        typename std::enable_if_t<!isComplex<_data_t>, _data_t> min() const
        {
            // TODO: dispatch to DataHandler backend and use optimized variants
            data_t min = std::numeric_limits<data_t>::max();
            for (index_t idx = 0; idx < this->getSize(); ++idx) {
                auto&& elem = (*this)[idx];
                if (elem < min) {
                    min = elem;
                }
            }
            return min;
        }

        /// return the maximum element of all the stored values
        template <typename _data_t = data_t>
        typename std::enable_if_t<!isComplex<_data_t>, _data_t> max() const
        {
            // TODO: dispatch to DataHandler backend and use optimized variants
            data_t max = std::numeric_limits<data_t>::min();
            for (index_t idx = 0; idx < this->getSize(); ++idx) {
                auto&& elem = (*this)[idx];
                if (elem > max) {
                    max = elem;
                }
            }
            return max;
        }

        /// compute in-place element-wise addition of another container
        DataContainer<data_t>& operator+=(const DataContainer<data_t>& dc);

        /// compute in-place element-wise addition with another expression
        template <typename Source, typename = std::enable_if_t<isExpression<Source>>>
        DataContainer<data_t>& operator+=(Source const& source)
        {
            *this = *this + source;
            return *this;
        }

        /// compute in-place element-wise subtraction of another container
        DataContainer<data_t>& operator-=(const DataContainer<data_t>& dc);

        /// compute in-place element-wise subtraction with another expression
        template <typename Source, typename = std::enable_if_t<isExpression<Source>>>
        DataContainer<data_t>& operator-=(Source const& source)
        {
            *this = *this - source;
            return *this;
        }

        /// compute in-place element-wise multiplication with another container
        DataContainer<data_t>& operator*=(const DataContainer<data_t>& dc);

        /// compute in-place element-wise multiplication with another expression
        template <typename Source, typename = std::enable_if_t<isExpression<Source>>>
        DataContainer<data_t>& operator*=(Source const& source)
        {
            *this = *this * source;
            return *this;
        }

        /// compute in-place element-wise division by another container
        DataContainer<data_t>& operator/=(const DataContainer<data_t>& dc);

        /// compute in-place element-wise division with another expression
        template <typename Source, typename = std::enable_if_t<isExpression<Source>>>
        DataContainer<data_t>& operator/=(Source const& source)
        {
            *this = *this / source;
            return *this;
        }

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

        /// @overload non-canst/read-write overload
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
        void format(std::ostream& os) const;

        /**
         * @brief Factory function which returns GPU based DataContainers
         *
         * @return the GPU based DataContainer
         *
         * Note that if this function is called on a container which is already GPU based, it will
         * throw an exception.
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

        /// helper function to get either the real or imaginary part
        /// from the complex values
        template <bool get_real, typename _data_t = data_t>
        typename std::enable_if_t<isComplex<_data_t>,
                                  DataContainer<GetFloatingPointType_t<_data_t>>>
            getComplexSplitup() const
        {
            using f_type = GetFloatingPointType_t<_data_t>;
            DataContainer<f_type> ret{
                *this->_dataDescriptor,
                this->_dataHandlerType,
            };

            // drop one of real/imaginary parts
            for (index_t idx = 0; idx < this->getSize(); ++idx) {
                auto&& val = (*this)[idx];
                if constexpr (get_real) {
                    ret[idx] = val.real();
                } else {
                    ret[idx] = val.imag();
                }
            }

            return ret;
        }
    };

    /// pretty output formatting.
    /// for configurable output, use `DataContainerFormatter` directly.
    template <typename T>
    std::ostream& operator<<(std::ostream& os, const elsa::DataContainer<T>& dc)
    {
        dc.format(os);
        return os;
    }

    /// Concatenate two DataContainers to one (requires copying of both)
    template <typename data_t>
    DataContainer<data_t> concatenate(const DataContainer<data_t>& dc1,
                                      const DataContainer<data_t>& dc2);

    /// User-defined template argument deduction guide for the expression based constructor
    template <typename Source>
    DataContainer(Source const& source) -> DataContainer<typename Source::data_t>;

    /// Collects callable lambdas for later dispatch
    template <typename... Ts>
    struct Callables : Ts... {
        using Ts::operator()...;
    };

    /// Class template deduction guide
    template <typename... Ts>
    Callables(Ts...) -> Callables<Ts...>;

    /// Multiplying two operands (including scalars)
    template <typename LHS, typename RHS, typename = std::enable_if_t<isBinaryOpOk<LHS, RHS>>>
    auto operator*(LHS const& lhs, RHS const& rhs)
    {
        auto multiplicationGPU = [](auto const& left, auto const& right, bool /**/) {
            return left * right;
        };

        if constexpr (isDcOrExpr<LHS> && isDcOrExpr<RHS>) {
            auto multiplication = [](auto const& left, auto const& right) {
                return (left.array() * right.array()).matrix();
            };
            return Expression{Callables{multiplication, multiplicationGPU}, lhs, rhs};
        } else if constexpr (isArithmetic<LHS>) {
            auto multiplication = [](auto const& left, auto const& right) {
                return (left * right.array()).matrix();
            };
            return Expression{Callables{multiplication, multiplicationGPU}, lhs, rhs};
        } else if constexpr (isArithmetic<RHS>) {
            auto multiplication = [](auto const& left, auto const& right) {
                return (left.array() * right).matrix();
            };
            return Expression{Callables{multiplication, multiplicationGPU}, lhs, rhs};
        } else {
            auto multiplication = [](auto const& left, auto const& right) { return left * right; };
            return Expression{Callables{multiplication, multiplicationGPU}, lhs, rhs};
        }
    }

    /// Adding two operands (including scalars)
    template <typename LHS, typename RHS, typename = std::enable_if_t<isBinaryOpOk<LHS, RHS>>>
    auto operator+(LHS const& lhs, RHS const& rhs)
    {
        auto additionGPU = [](auto const& left, auto const& right, bool /**/) {
            return left + right;
        };

        if constexpr (isDcOrExpr<LHS> && isDcOrExpr<RHS>) {
            auto addition = [](auto const& left, auto const& right) { return left + right; };
            return Expression{Callables{addition, additionGPU}, lhs, rhs};
        } else if constexpr (isArithmetic<LHS>) {
            auto addition = [](auto const& left, auto const& right) {
                return (left + right.array()).matrix();
            };
            return Expression{Callables{addition, additionGPU}, lhs, rhs};
        } else if constexpr (isArithmetic<RHS>) {
            auto addition = [](auto const& left, auto const& right) {
                return (left.array() + right).matrix();
            };
            return Expression{Callables{addition, additionGPU}, lhs, rhs};
        } else {
            auto addition = [](auto const& left, auto const& right) { return left + right; };
            return Expression{Callables{addition, additionGPU}, lhs, rhs};
        }
    }

    /// Subtracting two operands (including scalars)
    template <typename LHS, typename RHS, typename = std::enable_if_t<isBinaryOpOk<LHS, RHS>>>
    auto operator-(LHS const& lhs, RHS const& rhs)
    {
        auto subtractionGPU = [](auto const& left, auto const& right, bool /**/) {
            return left - right;
        };

        if constexpr (isDcOrExpr<LHS> && isDcOrExpr<RHS>) {
            auto subtraction = [](auto const& left, auto const& right) { return left - right; };
            return Expression{Callables{subtraction, subtractionGPU}, lhs, rhs};
        } else if constexpr (isArithmetic<LHS>) {
            auto subtraction = [](auto const& left, auto const& right) {
                return (left - right.array()).matrix();
            };
            return Expression{Callables{subtraction, subtractionGPU}, lhs, rhs};
        } else if constexpr (isArithmetic<RHS>) {
            auto subtraction = [](auto const& left, auto const& right) {
                return (left.array() - right).matrix();
            };
            return Expression{Callables{subtraction, subtractionGPU}, lhs, rhs};
        } else {
            auto subtraction = [](auto const& left, auto const& right) { return left - right; };
            return Expression{Callables{subtraction, subtractionGPU}, lhs, rhs};
        }
    }

    /// Dividing two operands (including scalars)
    template <typename LHS, typename RHS, typename = std::enable_if_t<isBinaryOpOk<LHS, RHS>>>
    auto operator/(LHS const& lhs, RHS const& rhs)
    {
        auto divisionGPU = [](auto const& left, auto const& right, bool /**/) {
            return left / right;
        };

        if constexpr (isDcOrExpr<LHS> && isDcOrExpr<RHS>) {
            auto division = [](auto const& left, auto const& right) {
                return (left.array() / right.array()).matrix();
            };
            return Expression{Callables{division, divisionGPU}, lhs, rhs};
        } else if constexpr (isArithmetic<LHS>) {
            auto division = [](auto const& left, auto const& right) {
                return (left / right.array()).matrix();
            };
            return Expression{Callables{division, divisionGPU}, lhs, rhs};
        } else if constexpr (isArithmetic<RHS>) {
            auto division = [](auto const& left, auto const& right) {
                return (left.array() / right).matrix();
            };
            return Expression{Callables{division, divisionGPU}, lhs, rhs};
        } else {
            auto division = [](auto const& left, auto const& right) { return left / right; };
            return Expression{Callables{division, divisionGPU}, lhs, rhs};
        }
    }

    /// Element-wise maximum value operation between two operands
    template <typename LHS, typename RHS, typename = std::enable_if_t<isBinaryOpOk<LHS, RHS>>>
    auto cwiseMax(LHS const& lhs, RHS const& rhs)
    {
        constexpr bool isLHSComplex = isComplex<GetOperandDataType_t<LHS>>;
        constexpr bool isRHSComplex = isComplex<GetOperandDataType_t<RHS>>;

#ifdef ELSA_CUDA_VECTOR
        auto cwiseMaxGPU = [](auto const& lhs, auto const& rhs, bool) {
            return quickvec::cwiseMax(lhs, rhs);
        };
#endif
        auto cwiseMax = [] {
            if constexpr (isLHSComplex && isRHSComplex) {
                return [](auto const& left, auto const& right) {
                    return (left.array().abs().max(right.array().abs())).matrix();
                };
            } else if constexpr (isLHSComplex) {
                return [](auto const& left, auto const& right) {
                    return (left.array().abs().max(right.array())).matrix();
                };
            } else if constexpr (isRHSComplex) {
                return [](auto const& left, auto const& right) {
                    return (left.array().max(right.array().abs())).matrix();
                };
            } else {
                return [](auto const& left, auto const& right) {
                    return (left.array().max(right.array())).matrix();
                };
            }
        }();

#ifdef ELSA_CUDA_VECTOR
        return Expression{Callables{cwiseMax, cwiseMaxGPU}, lhs, rhs};
#else
        return Expression{cwiseMax, lhs, rhs};
#endif
    }

    /// Element-wise absolute value operation
    template <typename Operand, typename = std::enable_if_t<isDcOrExpr<Operand>>>
    auto cwiseAbs(Operand const& operand)
    {
        auto abs = [](auto const& operand) { return (operand.array().abs()).matrix(); };
#ifdef ELSA_CUDA_VECTOR
        auto absGPU = [](auto const& operand, bool) { return quickvec::cwiseAbs(operand); };
        return Expression{Callables{abs, absGPU}, operand};
#else
        return Expression{abs, operand};
#endif
    }

    /// Element-wise square operation
    template <typename Operand, typename = std::enable_if_t<isDcOrExpr<Operand>>>
    auto square(Operand const& operand)
    {
        auto square = [](auto const& operand) { return (operand.array().square()).matrix(); };
#ifdef ELSA_CUDA_VECTOR
        auto squareGPU = [](auto const& operand, bool /**/) { return quickvec::square(operand); };
        return Expression{Callables{square, squareGPU}, operand};
#else
        return Expression{square, operand};
#endif
    }

    /// Element-wise square-root operation
    template <typename Operand, typename = std::enable_if_t<isDcOrExpr<Operand>>>
    auto sqrt(Operand const& operand)
    {
        auto sqrt = [](auto const& operand) { return (operand.array().sqrt()).matrix(); };
#ifdef ELSA_CUDA_VECTOR
        auto sqrtGPU = [](auto const& operand, bool /**/) { return quickvec::sqrt(operand); };
        return Expression{Callables{sqrt, sqrtGPU}, operand};
#else
        return Expression{sqrt, operand};
#endif
    }

    /// Element-wise exponenation operation with euler base
    template <typename Operand, typename = std::enable_if_t<isDcOrExpr<Operand>>>
    auto exp(Operand const& operand)
    {
        auto exp = [](auto const& operand) { return (operand.array().exp()).matrix(); };
#ifdef ELSA_CUDA_VECTOR
        auto expGPU = [](auto const& operand, bool /**/) { return quickvec::exp(operand); };
        return Expression{Callables{exp, expGPU}, operand};
#else
        return Expression{exp, operand};
#endif
    }

    /// Element-wise natural logarithm
    template <typename Operand, typename = std::enable_if_t<isDcOrExpr<Operand>>>
    auto log(Operand const& operand)
    {
        auto log = [](auto const& operand) { return (operand.array().log()).matrix(); };
#ifdef ELSA_CUDA_VECTOR
        auto logGPU = [](auto const& operand, bool /**/) { return quickvec::log(operand); };
        return Expression{Callables{log, logGPU}, operand};
#else
        return Expression{log, operand};
#endif
    }
} // namespace elsa
