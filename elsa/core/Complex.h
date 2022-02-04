#pragma once

#include <Eigen/Core>
#ifdef ELSA_CUDA_VECTOR
#include <thrust/complex.h>
#endif

#include <complex>

#ifdef ELSA_CUDA_VECTOR
// extend the thrust namespace by some standard mathematical functions
// this is required for the usage of thrust::complex with Eigen, and can be useful to us too
namespace thrust
{
    template <typename T>
    inline T real(const complex<T>& z)
    {
        return z.real();
    }

    template <typename T>
    inline T imag(const complex<T>& z)
    {
        return z.imag();
    }

    // template <typename T>
    // inline T abs2(const complex<T>& z)
    // {
    //     return real(z) * real(z) + imag(z) * imag(z);
    // }
} // namespace thrust
#endif

namespace elsa
{
#ifdef ELSA_CUDA_VECTOR
    template <typename T>
    using complex = thrust::complex<T>;

    using thrust::sqrt;
#else
    template <typename T>
    using complex = std::complex<T>;
#endif

    using std::sqrt;
} // namespace elsa

// add Eigen support for elsa::complex
namespace Eigen
{
#ifdef ELSA_CUDA_VECTOR
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
#endif
} // namespace Eigen
