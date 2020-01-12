#pragma once

#include "elsaDefines.h"
#include "DataDescriptor.h"
#include "DataHandler.h"
#include "DataHandlerCPU.h"
#include "DataHandlerMapCPU.h"
#include "DataContainerIterator.h"
#include "Expression.h"

#include <memory>
#include <type_traits>

namespace elsa
{

    /**
     * \brief class representing and storing a linearized n-dimensional signal
     *
     * \author Matthias Wieczorek - initial code
     * \author Tobias Lasser - rewrite, modularization, modernization
     * \author David Frank - added DataHandler concept, iterators
     * \author Nikola Dinev - add block support
     * \author Jens Petit - expression templates
     *
     * \tparam data_t - data type that is stored in the DataContainer, defaulting to real_t.
     *
     * This class provides a container for a signal that is stored in memory. This signal can
     * be n-dimensional, and will be stored in memory in a linearized fashion. The information
     * on how this linearization is performed is provided by an associated DataDescriptor.
     */
    template <typename data_t>
    class DataContainer
    {
    public:
        /// union of all possible handler raw pointers
        using HandlerTypes_t = std::variant<DataHandlerCPU<data_t>*, DataHandlerMapCPU<data_t>*>;

        /// delete default constructor (without metadata there can be no valid container)
        DataContainer() = delete;

        /**
         * \brief Constructor for empty DataContainer, initializing the data with zeros.
         *
         * \param[in] dataDescriptor containing the associated metadata
         * \param[in] handlerType the data handler (default: CPU)
         */
        explicit DataContainer(const DataDescriptor& dataDescriptor,
                               DataHandlerType handlerType = DataHandlerType::CPU);

        /**
         * \brief Constructor for DataContainer, initializing it with a DataVector
         *
         * \param[in] dataDescriptor containing the associated metadata
         * \param[in] data vector containing the initialization data
         * \param[in] handlerType the data handler (default: CPU)
         */
        DataContainer(const DataDescriptor& dataDescriptor,
                      const Eigen::Matrix<data_t, Eigen::Dynamic, 1>& data,
                      DataHandlerType handlerType = DataHandlerType::CPU);

        /**
         * \brief Copy constructor for DataContainer
         *
         * \param[in] other DataContainer to copy
         */
        DataContainer(const DataContainer<data_t>& other);

        /**
         * \brief copy assignment for DataContainer
         *
         * \param[in] other DataContainer to copy
         */
        DataContainer<data_t>& operator=(const DataContainer<data_t>& other);

        /**
         * \brief Move constructor for DataContainer
         *
         * \param[in] other DataContainer to move from
         *
         * The moved-from objects remains in a valid state. However, as preconditions are not
         * fulfilled for any member functions, the object should not be used. After move- or copy-
         * assignment, this is possible again.
         */
        DataContainer(DataContainer<data_t>&& other) noexcept;

        /**
         * \brief Move assignment for DataContainer
         *
         * \param[in] other DataContainer to move from
         *
         * The moved-from objects remains in a valid state. However, as preconditions are not
         * fulfilled for any member functions, the object should not be used. After move- or copy-
         * assignment, this is possible again.
         */
        DataContainer<data_t>& operator=(DataContainer<data_t>&& other);

        /**
         * \brief Expression evaluation assignment for DataContainer
         *
         * \param[in] source expression to evaluate
         *
         * This evaluates an expression template term into the underlying data member of
         * the DataHandler in use.
         */
        template <typename Source, typename = std::enable_if_t<isExpression<Source>>>
        DataContainer<data_t>& operator=(Source const& source)
        {
            _dataHandler->accessData() = source.eval();

            return *this;
        }

        /**
         * \brief Expression constructor
         *
         * \param[in] source expression to evaluate
         *
         * It creates a new DataContainer out of an expression. For this the meta information which
         * is saved in the expression is used.
         */
        template <typename Source, typename = std::enable_if_t<isExpression<Source>>>
        DataContainer<data_t>(Source const& source)
            : DataContainer<data_t>(source.getDataMetaInfo().first, source.eval(),
                                    source.getDataMetaInfo().second)
        {
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
            return (*this * source).eval().sum();
        }

        /// return the squared l2 norm of this signal (dot product with itself)
        data_t squaredL2Norm() const;

        /// return the l1 norm of this signal (sum of absolute values)
        data_t l1Norm() const;

        /// return the linf norm of this signal (maximum of absolute values)
        data_t lInfNorm() const;

        /// return the sum of all elements of this signal
        data_t sum() const;

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
        template <class Operand, std::enable_if_t<isDataContainer<Operand>, int>>
        friend constexpr auto evaluateOrReturn(Operand const& operand);

    private:
        /// returns the underlying derived handler as a raw pointer in a std::variant
        HandlerTypes_t getHandlerPtr() const;

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
                               DataHandlerType dataType = DataHandlerType::CPU);
    };

    /// User-defined template argument deduction guide for the expression based constructor
    template <typename Source>
    DataContainer(Source const& source)->DataContainer<typename Source::data_t>;

    /// Multiplying two operands (including scalars)
    template <typename LHS, typename RHS, typename = std::enable_if_t<isBinaryOpOk<LHS, RHS>>>
    auto operator*(LHS const& lhs, RHS const& rhs)
    {
        if constexpr (isDcOrExpr<LHS> && isDcOrExpr<RHS>) {
            auto multiplication = [](auto const& left, auto const& right) {
                return (left.array() * right.array()).matrix();
            };
            return Expression{multiplication, lhs, rhs};
        } else if constexpr (isArithmetic<LHS>) {
            auto multiplication = [](auto const& left, auto const& right) {
                return (left * right.array()).matrix();
            };
            return Expression{multiplication, lhs, rhs};
        } else if constexpr (isArithmetic<RHS>) {
            auto multiplication = [](auto const& left, auto const& right) {
                return (left.array() * right).matrix();
            };
            return Expression{multiplication, lhs, rhs};
        } else {
            auto multiplication = [](auto const& left, auto const& right) { return left * right; };
            return Expression{multiplication, lhs, rhs};
        }
    }

    /// Adding two operands (including scalars)
    template <typename LHS, typename RHS, typename = std::enable_if_t<isBinaryOpOk<LHS, RHS>>>
    auto operator+(LHS const& lhs, RHS const& rhs)
    {
        if constexpr (isDcOrExpr<LHS> && isDcOrExpr<RHS>) {
            auto addition = [](auto const& left, auto const& right) { return left + right; };
            return Expression{addition, lhs, rhs};
        } else if constexpr (isArithmetic<LHS>) {
            auto addition = [](auto const& left, auto const& right) {
                return (left + right.array()).matrix();
            };
            return Expression{addition, lhs, rhs};
        } else if constexpr (isArithmetic<RHS>) {
            auto addition = [](auto const& left, auto const& right) {
                return (left.array() + right).matrix();
            };
            return Expression{addition, lhs, rhs};
        } else {
            auto addition = [](auto const& left, auto const& right) { return left + right; };
            return Expression{addition, lhs, rhs};
        }
    }

    /// Subtracting two operands (including scalars)
    template <typename LHS, typename RHS, typename = std::enable_if_t<isBinaryOpOk<LHS, RHS>>>
    auto operator-(LHS const& lhs, RHS const& rhs)
    {
        if constexpr (isDcOrExpr<LHS> && isDcOrExpr<RHS>) {
            auto subtraction = [](auto const& left, auto const& right) { return left - right; };
            return Expression{subtraction, lhs, rhs};
        } else if constexpr (isArithmetic<LHS>) {
            auto subtraction = [](auto const& left, auto const& right) {
                return (left - right.array()).matrix();
            };
            return Expression{subtraction, lhs, rhs};
        } else if constexpr (isArithmetic<RHS>) {
            auto subtraction = [](auto const& left, auto const& right) {
                return (left.array() - right).matrix();
            };
            return Expression{subtraction, lhs, rhs};
        } else {
            auto subtraction = [](auto const& left, auto const& right) { return left - right; };
            return Expression{subtraction, lhs, rhs};
        }
    }

    /// Dividing two operands (including scalars)
    template <typename LHS, typename RHS, typename = std::enable_if_t<isBinaryOpOk<LHS, RHS>>>
    auto operator/(LHS const& lhs, RHS const& rhs)
    {
        if constexpr (isDcOrExpr<LHS> && isDcOrExpr<RHS>) {
            auto division = [](auto const& left, auto const& right) {
                return (left.array() / right.array()).matrix();
            };
            return Expression{division, lhs, rhs};
        } else if constexpr (isArithmetic<LHS>) {
            auto division = [](auto const& left, auto const& right) {
                return (left / right.array()).matrix();
            };
            return Expression{division, lhs, rhs};
        } else if constexpr (isArithmetic<RHS>) {
            auto division = [](auto const& left, auto const& right) {
                return (left.array() / right).matrix();
            };
            return Expression{division, lhs, rhs};
        } else {
            auto division = [](auto const& left, auto const& right) { return left * right; };
            return Expression{division, lhs, rhs};
        }
    }

    /// Element-wise square operation
    template <typename Operand, typename = std::enable_if_t<isDcOrExpr<Operand>>>
    auto square(Operand const& operand)
    {
        auto square = [](auto const& operand) { return (operand.array().square()).matrix(); };
        return Expression{square, operand};
    }

    /// Element-wise square-root operation
    template <typename Operand, typename = std::enable_if_t<isDcOrExpr<Operand>>>
    auto sqrt(Operand const& operand)
    {
        auto sqrt = [](auto const& operand) { return (operand.array().sqrt()).matrix(); };
        return Expression{sqrt, operand};
    }

    /// Element-wise exponenation operation with euler base
    template <typename Operand, typename = std::enable_if_t<isDcOrExpr<Operand>>>
    auto exp(Operand const& operand)
    {
        auto exp = [](auto const& operand) { return (operand.array().exp()).matrix(); };
        return Expression{exp, operand};
    }

    /// Element-wise natural logarithm
    template <typename Operand, typename = std::enable_if_t<isDcOrExpr<Operand>>>
    auto log(Operand const& operand)
    {
        auto log = [](auto const& operand) { return (operand.array().log()).matrix(); };
        return Expression{log, operand};
    }

} // namespace elsa
