#pragma once

#include <complex>
#include <cstddef>
#include <Eigen/Core>
#include <type_traits>

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

    /// global type for vectors of data_t
    template <typename data_t>
    using Vector_t = Eigen::Matrix<data_t, Eigen::Dynamic, 1>;

    /// global type for matrices of real numbers
    using RealMatrix_t = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic>;

    /// template global constexpr for the number pi
    template <typename T>
    constexpr auto pi = static_cast<T>(3.14159265358979323846);

    /// global constexpr for the number pi
    constexpr auto pi_t = pi<real_t>;

    /// type of the DataHandler used to store the actual data
    enum class DataHandlerType {
        CPU,     ///< data is stored as an Eigen::Matrix in CPU main memory
        MAP_CPU, ///< data is not explicitly stored, but using an Eigen::Map to refer to other
        GPU,     ///< data is stored as an raw array in the GPU memory
        MAP_GPU  ///< data is not explicitley stored but mapped through a pointer
    };

#ifdef ELSA_CUDA_VECTOR
    constexpr DataHandlerType defaultHandlerType = DataHandlerType::GPU;
#else
    constexpr DataHandlerType defaultHandlerType = DataHandlerType::CPU;
#endif

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
    constexpr bool isComplex = std::is_same<RemoveCvRef_t<T>, std::complex<float>>::value
                               || std::is_same<RemoveCvRef_t<T>, std::complex<double>>::value;

    /**
     * Helper function to be used as failure case with constexpr-ifs:
     *
     * if constexpr (bla) {
     *     ...
     * }
     * else {
     *     match_failure();
     * }
     */
    template <bool nope = false>
    void branch_match_failure()
    {
        static_assert(nope, "no static branch match found");
    }
} // namespace elsa

/*
 * Branch prediction tuning.
 * the expression is expected to be true (=likely) or false (=unlikely).
 *
 * btw, this implementation was taken from the Linux kernel.
 */
#if defined(__GNUC__)
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif
