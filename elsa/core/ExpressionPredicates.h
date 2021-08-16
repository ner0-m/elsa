#pragma once

#include "elsaDefines.h"
#include <type_traits>

namespace elsa
{
    /// User defined is_arithmetic which includes complex numbers
    template <typename T>
    constexpr bool isArithmetic = std::is_arithmetic_v<T> || isComplex<T>;

    /// forward declaration for predicates and friend test function
    template <typename data_t = real_t>
    class DataContainer;

    /// Base case inheriting false
    template <typename>
    struct IsDataContainerType : std::false_type {
    };

    /// Partial specialization which inherits true
    template <typename data_t>
    struct IsDataContainerType<DataContainer<data_t>> : std::true_type {
    };

    /// Predicate to check Operand
    template <class T>
    constexpr bool isDataContainer = IsDataContainerType<RemoveCvRef_t<T>>();

    /// Forward declaration for predicates
    template <typename Callable, typename... Operands>
    class Expression;

    /// Base case inheriting false
    template <typename>
    struct IsExpressionType : std::false_type {
    };

    /// Partial specialization inheriting true
    template <typename Callable, typename... Operands>
    struct IsExpressionType<Expression<Callable, Operands...>> : std::true_type {
    };

    /// Predicate to check operand
    template <class T>
    constexpr bool isExpression = IsExpressionType<RemoveCvRef_t<T>>();

    /// Predicate to check operand
    template <class T>
    constexpr bool isDcOrExpr = isDataContainer<T> || isExpression<T>;

    /// Combined predicate for binary operations
    template <typename LHS, typename RHS>
    constexpr bool isBinaryOpOk = (isDcOrExpr<LHS> && isDcOrExpr<RHS>)
                                  || (isDcOrExpr<LHS> && isArithmetic<RHS>)
                                  || (isArithmetic<LHS> && isDcOrExpr<RHS>);

    /// Default case to infer data_t of any operand
    template <typename Operand>
    struct GetOperandDataType {
        using data_t = real_t;
    };

    /// Partial specialization to infer data_t from DataContainer
    template <typename data_type>
    struct GetOperandDataType<DataContainer<data_type>> {
        using data_t = data_type;
    };

    /// Partial specialization to infer data_t from Expression
    template <typename Callable, typename... Operands>
    struct GetOperandDataType<Expression<Callable, Operands...>> {
        using data_t = typename Expression<Callable, Operands...>::data_t;
    };

    template <typename T>
    using GetOperandDataType_t = typename GetOperandDataType<T>::data_t;

    /* Uses the data type used in the first or last operand depending on whether the first operand
     * is an anrithmetic type
     */
    template <typename... Operands>
    struct GetOperandsDataType {
        using data_t = std::conditional_t<
            isArithmetic<std::tuple_element_t<0, std::tuple<Operands...>>>,
            std::tuple_element_t<sizeof...(Operands) - 1,
                                 std::tuple<typename GetOperandDataType<Operands>::data_t...>>,
            std::tuple_element_t<0, std::tuple<typename GetOperandDataType<Operands>::data_t...>>>;
    };

} // namespace elsa
