#pragma once

#include <thrust/complex.h>

#include <complex>

namespace elsa
{
    template <typename T>
    using complex = thrust::complex<T>;

    using std::sqrt;
    using thrust::sqrt;
} // namespace elsa

namespace std
{
    template <class T, class U>
    struct common_type<thrust::complex<T>, thrust::complex<U>> {
        using type = thrust::complex<std::common_type_t<T, U>>;
    };
} // namespace std

#include <Eigen/Core>

// add Eigen support for elsa::complex
namespace Eigen
{
    // std::complex is supported out-of-the-box, so we only care about thrust::complex
    template <typename T>
    struct NumTraits<thrust::complex<T>> : NumTraits<std::complex<T>> {
        typedef T Real;
        typedef thrust::complex<T> NonInteger;
        typedef thrust::complex<T> Nested;
        enum {
            IsComplex = 1,
            IsInteger = 0,
            IsSigned = 1,
            RequireInitialization = NumTraits<Real>::RequireInitialization,
            ReadCost = 2 * NumTraits<Real>::ReadCost,
            AddCost = 2 * NumTraits<Real>::AddCost,
            MulCost = 4 * NumTraits<Real>::MulCost + 2 * NumTraits<Real>::AddCost
        };
    };
} // namespace Eigen
