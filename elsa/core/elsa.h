#pragma once

#include <complex>
#include <cstddef>
#include <Eigen/Core>

namespace elsa
{
    using real_t = float; ///< global type for real numbers
    using complex_t = std::complex<real_t>; ///< global type for complex numbers
    using index_t = std::ptrdiff_t ; ///< global type for indices

    using RealVector_t = Eigen::Matrix<real_t, Eigen::Dynamic, 1>; ///< global type for vectors of real numbers
    using ComplexVector_t = Eigen::Matrix<complex_t, Eigen::Dynamic, 1>; ///< global type for vectors of complex numbers
    using IndexVector_t = Eigen::Matrix<index_t, Eigen::Dynamic, 1>; ///< global type for vectors of indices
    using BooleanVector_t = Eigen::Matrix<bool, Eigen::Dynamic, 1>; ///< global type for vectors of booleans

    using RealMatrix_t = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic>; ///< global type for matrices of real numbers

    constexpr double pi = 3.14159265358979323846; ///< global constexpr for the number pi
} // namespace elsa
