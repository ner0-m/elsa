#pragma once

#include "elsaDefines.h"
#include <type_traits>

namespace elsa
{
    /// Remove cv qualifiers as well as reference of given type
    // TODO: Replace with std::remove_cv_ref_t when C++20 available
    template <typename T>
    struct RemoveCvRef {
        using type = std::remove_cv_t<std::remove_reference_t<T>>;
    };

    /// Helper to make type available
    template <class T>
    using RemoveCvRef_t = typename RemoveCvRef<T>::type;

    /// Predicate to check if of complex type
    template <typename T>
    constexpr bool isComplex =
        std::is_same_v<
            RemoveCvRef_t<T>,
            std::complex<float>> || std::is_same_v<RemoveCvRef_t<T>, std::complex<double>>;

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
} // namespace elsa
