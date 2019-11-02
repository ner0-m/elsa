#pragma once

#include <tuple>
#include <vector>
#include <iostream>
#include "elsa.h"

namespace elsa
{
    struct BaseDataContainer {
    };
    // forward declaration for friend test function
    template <typename data_t = real_t>
    class DataContainer;

    // Predicate to check Operand
    template <class T>
    constexpr bool isDataContainer =
        std::is_base_of_v<BaseDataContainer, std::remove_cv_t<std::remove_reference_t<T>>>;

    // this dummy base class is needed as all derived template instantiations of Expressions have to
    // be dealt with
    struct BaseExpression {
    };

    // Predicate to check Operand
    template <class T>
    constexpr bool isExpression =
        std::is_base_of_v<BaseExpression, std::remove_cv_t<std::remove_reference_t<T>>>;

    // Predicate to check Operand
    template <class T>
    constexpr bool isDcOrExpr = isDataContainer<T> || isExpression<T>;

    // Predicate to check Operand
    template <typename LHS, typename RHS>
    constexpr bool isBinaryOpOk = isDcOrExpr<LHS> || isDcOrExpr<RHS>;

    // compile time switch to select if recursively evaluate or not for each operand
    template <class Operand>
    constexpr auto evaluateOrReturn(Operand const& v, index_t const i)
    {
        if constexpr (isDcOrExpr<Operand>) {
            return v[i];
        } else {
            return v;
        }
    }

    template <typename T>
    class ReferenceOrNot {
    public:
        using type = typename std::conditional<isExpression<T>, T, const T&>::type;
    };

    // The Callable takes the Operands
    template <typename Callable, typename... Operands>
    class Expression : public BaseExpression
    {

    private:
        Callable _callable;

        // expression types are saved as copies, whereas all datacontainers only as references
        std::tuple<typename ReferenceOrNot<Operands>::type...> _args;

    public:
        Expression(Callable func, Operands const&... args) : _callable(func), _args(args...) {}

        // this operator evaluates the expression
        auto operator[](index_t i) const
        {

            // generic lambda for evaluating tree, we need this to get a pack again out of the tuple
            auto const callAtIndex = [this, i](Operands const&... args) {
                // here evaluateOrReturn is called on each Operand within args as the unpacking
                // takes place after the fcn call
                return _callable(evaluateOrReturn(args, i)...);
            };
            return std::apply(callAtIndex, _args);
        }
    };

} // namespace elsa
