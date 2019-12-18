#pragma once

#include "elsaDefines.h"
#include <tuple>
#include <variant>
#include <optional>
#include "DataDescriptor.h"
#include "ExpressionPredicates.h"

namespace elsa
{
    /// Compile time switch to select to recursively evaluate for expression type
    template <class Operand, std::enable_if_t<isExpression<Operand>, int> = 0>
    constexpr auto evaluateOrReturn(Operand const& operand)
    {
        return operand.eval();
    }

    /// Compile time switch to return-by-value of arithmetic types
    template <class Operand, std::enable_if_t<isArithmetic<Operand>, int> = 0>
    constexpr auto evaluateOrReturn(Operand const operand)
    {
        return operand;
    }

    /// Compile time switch to return data in container
    template <class Operand, std::enable_if_t<isDataContainer<Operand>, int> = 0>
    constexpr auto evaluateOrReturn(Operand const& operand)
    {
        return operand._dataHandler->accessData();
    }

    /// Type trait which decides if const lvalue reference or not
    template <typename Operand>
    class ReferenceOrNot
    {
    public:
        using type = typename std::conditional<isExpression<Operand> || isArithmetic<Operand>,
                                               Operand, const Operand&>::type;
    };

    /**
     * \brief Temporary expression type which enables lazy-evaluation of expression
     *
     * \author Jens Petit
     *
     * \tparam Callable - the operation to be performed
     * \tparam Operands - the objects on which the operation is performed
     *
     */
    template <typename Callable, typename... Operands>
    class Expression
    {
    public:
        /// type which bundles the meta information to create a new DataContainer
        using MetaInfo_t = std::pair<DataDescriptor const&, DataHandlerType>;

        /// Constructor
        Expression(Callable func, Operands const&... args)
            : _callable(func), _args(args...), _dataMetaInfo(initDescriptor(args...))
        {
        }

        /// Evaluates the expression
        auto eval() const
        {
            // generic lambda for evaluating tree, we need this to get a pack again out of the tuple
            auto const callCallable = [this](Operands const&... args) {
                // here evaluateOrReturn is called on each Operand within args as the unpacking
                // takes place after the fcn call
                return _callable(evaluateOrReturn(args)...);
            };
            return std::apply(callCallable, _args);
        }

        MetaInfo_t getDataMetaInfo() const { return _dataMetaInfo; }

    private:
        /// The function to call on the operand(s)
        const Callable _callable;

        /// Contains all operands saved as const references (DataContainers) or copies
        /// (Expressions and arithmetic types)
        std::tuple<typename ReferenceOrNot<Operands>::type...> _args;

        /// saves the meta information to create a new DataContainer out of an expression
        const MetaInfo_t _dataMetaInfo;

        /// correctly returns the DataContainer descriptor based on the operands (either
        /// expressions or Datacontainers)
        MetaInfo_t initDescriptor(Operands const&... args)
        {
            if (auto info = getMetaInfoFromContainers(args...); info.has_value()) {
                return *info;
            } else {
                if (auto info = getMetaInfoFromExpressions(args...); info.has_value()) {
                    return *info;
                } else {
                    throw std::logic_error("No meta info available, cannot create expression");
                }
            }
        }

        /// base recursive case if no DataContainer as operand
        std::optional<MetaInfo_t> getMetaInfoFromContainers() { return {}; }

        /// recursive traversal of all contained DataContainers
        template <class T, class... Ts>
        std::optional<MetaInfo_t> getMetaInfoFromContainers(T& arg, Ts&... args)
        {
            if constexpr (isDataContainer<T>) {
                return MetaInfo_t{arg.getDataDescriptor(), arg.getDataHandlerType()};
            } else {
                return getMetaInfoFromContainers(args...);
            }
        }

        /// base recursive case if no Expression as operand
        std::optional<MetaInfo_t> getMetaInfoFromExpressions() { return {}; }

        /// recursive traversal of all contained Expressions
        template <class T, class... Ts>
        std::optional<MetaInfo_t> getMetaInfoFromExpressions(T& arg, Ts&... args)
        {
            if constexpr (isExpression<T>) {
                return arg.getDataMetaInfo();
            } else {
                return getMetaInfoFromExpressions(args...);
            }
        }
    };
} // namespace elsa
