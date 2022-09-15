#pragma once

#include "CUDADefines.h"
#include <utility>
#include "thrust/complex.h"

namespace elsa
{
    // Arithmetic operations
    // negate
    struct negate {
        template <class T>
        __host__ __device__ constexpr auto operator()(const T& arg)
            -> decltype(-std::declval<T>()) const
        {
            return -arg;
        }
    };

    // identity
    struct identity {
        template <class T>
        __host__ __device__ constexpr auto operator()(const T& arg)
            -> decltype(std::declval<T>()) const
        {
            return arg;
        }
    };

    // plus
    struct plus {
        template <class T, class U = T>
        __host__ __device__ constexpr auto operator()(const T& lhs, const U& rhs)
            -> decltype(std::declval<T>() + std::declval<U>()) const
        {
            return lhs + rhs;
        }
    };

    // minus
    struct minus {
        template <class T, class U = T>
        __host__ __device__ constexpr auto operator()(const T& lhs, const U& rhs)
            -> decltype(std::declval<T>() - std::declval<U>()) const
        {
            return lhs - rhs;
        }
    };

    // multiplies
    struct multiplies {
        template <class T, class U = T>
        __host__ __device__ constexpr auto operator()(const T& lhs, const U& rhs)
            -> decltype(std::declval<T>() * std::declval<U>()) const
        {
            return lhs * rhs;
        }
    };

    // divides
    struct divides {
        template <class T, class U = T>
        __host__ __device__ constexpr auto operator()(const T& lhs, const U& rhs)
            -> decltype(std::declval<T>() / std::declval<U>()) const
        {
            return lhs / rhs;
        }
    };

    // modulus
    struct modulus {
        template <class T, class U = T>
        __host__ __device__ constexpr auto operator()(const T& lhs, const U& rhs)
            -> decltype(std::declval<T>() % std::declval<U>()) const
        {
            return lhs % rhs;
        }
    };
} // namespace elsa
