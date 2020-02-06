#pragma once

#include <complex>
#include <cstddef>
#include <Eigen/Core>

namespace elsa
{
    using real_t = float;                   ///< global type for real numbers
    using complex_t = std::complex<real_t>; ///< global type for complex numbers
    using index_t = std::ptrdiff_t;         ///< global type for indices

    /// global type for vectors of real numbers
    using RealVector_t = Eigen::Matrix<real_t, Eigen::Dynamic, 1>;

    /// global type for vectors of complex numbers
    using ComplexVector_t = Eigen::Matrix<complex_t, Eigen::Dynamic, 1>;

    /// global type for vectors of indices
    using IndexVector_t = Eigen::Matrix<index_t, Eigen::Dynamic, 1>;

    /// global type for vectors of booleans
    using BooleanVector_t = Eigen::Matrix<bool, Eigen::Dynamic, 1>;

    /// global type for matrices of real numbers
    using RealMatrix_t = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic>;

    /// template global constexpr for the number pi
    template <typename T>
    constexpr auto pi = static_cast<T>(3.14159265358979323846);

    /// global constexpr for the number pi
    constexpr auto pi_t = pi<real_t>;

    /// type of the DataHandler used to store the actual data
    enum class DataHandlerType {
        CPU,    ///< data is stored as an Eigen::Matrix in CPU main memory
        MAP_CPU ///< data is not explicitly stored, but using an Eigen::Map to refer to other
    };

    /// base case for deducing floating point type of std::complex
    template <typename T>
    struct GetFloatingPointType {
        using type = T;
    };

    /// partial specialization to derive correct floating point type
    template <typename T>
    struct GetFloatingPointType<std::complex<T>> {
        using type = T;
    };

    /// helper typedef to facilitate usage
    template <typename T>
    using GetFloatingPointType_t = typename GetFloatingPointType<T>::type;

} // namespace elsa
