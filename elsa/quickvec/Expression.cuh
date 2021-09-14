#pragma once

#include <tuple>
#include <functional>
#include <type_traits>
#include <cuda_runtime.h>
#include <thrust/complex.h>

#include "Defines.cuh"

namespace quickvec
{
    /// base case for deducing floating point type of std::complex
    template <typename T>
    struct GetFloatingPointType {
        using type = T;
    };

    /// partial specialization to derive correct floating point type
    template <typename T>
    struct GetFloatingPointType<thrust::complex<T>> {
        using type = T;
    };

    /// helper typedef to facilitate usage
    template <typename T>
    using GetFloatingPointType_t = typename GetFloatingPointType<T>::type;

    /// Remove cv qualifiers as well as reference of given type
    // TODO: Replace with std::remove_cv_ref_t when C++20 available
    template <typename T>
    struct RemoveCvRef {
        using type = std::remove_cv_t<std::remove_reference_t<T>>;
    };

    /// Helper to make type available
    // TODO: Replace with std::remove_cv_ref_t when C++20 available
    template <class T>
    using RemoveCvRef_t = typename RemoveCvRef<T>::type;

    /// Predicate to check if of complex type
    template <typename T>
    constexpr bool isComplex = std::is_same<RemoveCvRef_t<T>, thrust::complex<float>>::value
                               || std::is_same<RemoveCvRef_t<T>, thrust::complex<double>>::value;

    /// User defined is_arithmetic which includes complex numbers
    template <typename T>
    constexpr bool isArithmetic = std::is_arithmetic_v<T> || isComplex<T>;

    // forward declaration for predicates
    template <typename data_t = float>
    class Vector;

    /// Base case inheriting false
    template <typename>
    struct IsVectorType : std::false_type {
    };

    /// Partial specialization which inherits true
    template <typename data_t>
    struct IsVectorType<quickvec::Vector<data_t>> : std::true_type {
    };

    // Predicate to check Operand
    template <class T>
    constexpr bool isVector = IsVectorType<RemoveCvRef_t<T>>();

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

    /// Predicate to check Operand
    template <class T>
    constexpr bool isVectorOrExpression = isVector<T> || isExpression<T>;

    /// Predicate to check if a binary operation is ok
    template <typename LHS, typename RHS>
    constexpr bool isBinaryOpOk = (isVectorOrExpression<LHS> && isVectorOrExpression<RHS>)
                                  || (isVectorOrExpression<LHS> && isArithmetic<RHS>)
                                  || (isArithmetic<LHS> && isVectorOrExpression<RHS>);

    /// compile time switch to select if recursively evaluate or not for each operand
    template <class Operand>
    __device__ constexpr auto evaluateOrReturn(Operand const& operand, size_t const i)
    {
        if constexpr (isVectorOrExpression<Operand>) {
            // further traversing the tree if Expression or Vector
            return operand[i];
        } else {
            // for scalar case returning the scalar
            return operand;
        }
    }

    // The Callable takes the Operands
    template <typename Callable, typename... Operands>
    class Expression
    {
    private:
        /// defines the operation to do between the operands
        Callable _callable;

        /// everything is saved by value as we want later to copy everything to the device
        std::tuple<Operands...> _args;

    public:
        Expression(Callable func, Operands const&... args) : _callable(func), _args(args...) {}

        // operator evaluates the expression on the device
        __device__ auto operator[](size_t i) const
        {
            if constexpr (std::tuple_size_v<decltype(_args)> == 1) {
                return _callable(evaluateOrReturn(std::get<0>(_args), i));
            } else {
                return _callable(evaluateOrReturn(std::get<0>(_args), i),
                                 evaluateOrReturn(std::get<1>(_args), i));
            }
        }
    };
} // namespace quickvec
