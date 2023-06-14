#pragma once

#include "DisableWarnings.h"

#include <complex>
#include <cstddef>
#include <type_traits>

#include <Eigen/Core>
#include <Eigen/Geometry>

#ifdef ELSA_CUDA_VECTOR
#include <thrust/complex.h>
#endif

#include "Complex.h"

namespace elsa
{
    using real_t = float;              ///< global type for real numbers
    using complex_t = complex<real_t>; ///< global type for complex numbers
    using index_t = std::ptrdiff_t;    ///< global type for indices

    /// global type for vectors of real numbers
    using RealVector_t = Eigen::Matrix<real_t, Eigen::Dynamic, 1>;

    /// global type for vectors of complex numbers
    using ComplexVector_t = Eigen::Matrix<complex_t, Eigen::Dynamic, 1>;

    /// global type for vectors of indices
    using IndexVector_t = Eigen::Matrix<index_t, Eigen::Dynamic, 1>;

    /// global type for vectors of booleans
    using BooleanVector_t = Eigen::Matrix<bool, Eigen::Dynamic, 1>;

    /// global type for vectors of data_t
    template <typename data_t>
    using Vector_t = Eigen::Matrix<data_t, Eigen::Dynamic, 1>;

    /// global type for arrays of index_t of size dim
    template <int dim>
    using IndexArray_t = Eigen::Array<index_t, dim, 1>;

    /// global type for arrrays of real_t of size dim
    template <int dim>
    using RealArray_t = Eigen::Array<real_t, dim, 1>;

    /// global type for arrays of bol of size dim
    template <int dim>
    using BooleanArray_t = Eigen::Array<bool, dim, 1>;
    /// global type for matrices of real numbers
    using RealMatrix_t = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic>;

    /// global type alias for rays
    using RealRay_t = Eigen::ParametrizedLine<real_t, Eigen::Dynamic>;

    /// global type alias for rays
    template <typename data_t>
    using Ray_t = Eigen::ParametrizedLine<data_t, Eigen::Dynamic>;

    /// template global constexpr for the number pi
    template <typename T>
    constexpr auto pi = static_cast<T>(3.14159265358979323846);

    /// global constexpr for the number pi
    constexpr auto pi_t = pi<real_t>;

    /// various values of the different norms of the Fourier transforms
    enum class FFTNorm { FORWARD, ORTHO, BACKWARD };

    /// base case for deducing floating point type of std::complex
    template <typename T>
    struct GetFloatingPointType {
        using type = T;
    };

    /// partial specialization to derive correct floating point type
    template <typename T>
    struct GetFloatingPointType<complex<T>> {
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
    template <class T>
    using RemoveCvRef_t = typename RemoveCvRef<T>::type;

    /// Predicate to check if of complex type
    template <typename T>
    constexpr bool isComplex = std::is_same<RemoveCvRef_t<T>, complex<float>>::value
                               || std::is_same<RemoveCvRef_t<T>, complex<double>>::value;

    /// With C++20 this can be replaced by std::type_identity
    template <class T>
    struct SelfType {
        using type = T;
    };
    template <class T>
    using SelfType_t = typename SelfType<T>::type;
} // namespace elsa

/*
 * Branch prediction tuning.
 * the expression is expected to be true (=likely) or false (=unlikely).
 *
 * btw, this implementation was taken from the Linux kernel.
 */
#if defined(__GNUC__) || defined(__clang__)
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif
