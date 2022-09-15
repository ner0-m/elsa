#pragma once

#include <type_traits>

#include <complex>
#include "thrust/complex.h"

namespace elsa
{
    //*********************************************************************************************
    template <bool B, typename T = void>
    using disable_if = std::enable_if<!B, T>;

    template <bool B, typename T = void>
    using disable_if_t = typename disable_if<B, T>::type;
    //*********************************************************************************************

    //*********************************************************************************************
    // check if a type is a specialization of another, i.e. usage:
    // `is_specialization<T, std::vector>::value`
    template <typename Test, template <typename...> class Ref>
    struct is_specialization : std::false_type {
    };

    template <template <typename...> class Ref, typename... Args>
    struct is_specialization<Ref<Args...>, Ref> : std::true_type {
    };

    template <class T, template <class...> class Ref>
    inline constexpr bool is_specialization_v = is_specialization<T, Ref>::value;
    //*********************************************************************************************

    //*********************************************************************************************
    // value_type_of for complex types, i.e. returns T, for std::complex<T>
    template <typename T>
    struct value_type_of {
        using type = T;
    };

    template <typename T>
    struct value_type_of<std::complex<T>> {
        using type = typename std::complex<T>::value_type;
    };

    template <typename T>
    struct value_type_of<thrust::complex<T>> {
        using type = typename thrust::complex<T>::value_type;
    };

    template <typename T>
    using value_type_of_t = typename value_type_of<T>::type;
    //*********************************************************************************************

    //*********************************************************************************************
    // check if a type is std::complex or thrust::complex
    template <typename T>
    struct is_complex : std::false_type {
    };

    template <typename T>
    struct is_complex<std::complex<T>> : std::true_type {
    };

    template <typename T>
    struct is_complex<thrust::complex<T>> : std::true_type {
    };

    template <class T>
    constexpr bool is_complex_v = is_complex<T>::value;
    //*********************************************************************************************

    //*********************************************************************************************
    // Wrap T with thrust/std::complex, expect if it's already complex
    template <typename T>
    struct add_complex {
        using type = thrust::complex<T>;
    };

    template <typename T>
    struct add_complex<thrust::complex<T>> {
        using type = thrust::complex<T>;
    };

    template <typename T>
    struct add_complex<std::complex<T>> {
        using type = std::complex<T>;
    };

    template <class T>
    using add_complex_t = typename add_complex<T>::type;
    //*********************************************************************************************

    //*********************************************************************************************
    // Wrap T with thrust/std::complex, expect if it's already complex
    template <typename T>
    struct is_scalar : std::conditional_t<std::is_arithmetic_v<T> || is_complex_v<T>,
                                          std::true_type, std::false_type> {
    };

    template <typename T>
    inline constexpr bool is_scalar_v = is_scalar<T>::value;

    //*********************************************************************************************
} // namespace elsa
