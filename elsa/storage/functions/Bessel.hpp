#pragma once

#include "CUDADefines.h"
#include <cmath>

namespace elsa::fn
{
    namespace detail
    {
        struct BesselFn_log_0 {
            template <typename T>
            __host__ __device__ constexpr T operator()(const T& arg) const noexcept
            {
                if (arg < static_cast<T>(3.75)) {
                    const T ratio = arg / static_cast<T>(3.75);
                    const T y = ratio * ratio;

                    const T p0 = static_cast<T>(0.45813e-2);
                    const T p1 = static_cast<T>(0.360768e-1) + y * p0;
                    const T p2 = static_cast<T>(0.2659732) + y * p1;
                    const T p3 = static_cast<T>(1.2067492) + y * p2;
                    const T p4 = static_cast<T>(3.0899424) + y * p3;
                    const T p5 = static_cast<T>(3.5156229) + y * p4;
                    const T p6 = static_cast<T>(1.0) + y * p5;

                    return std::log(p6);
                } else {
                    // see Numerical Recipes in C - 2nd Edition
                    // by W. H. Press, S. A. Teukolsky, W. T. Vetterling, B. P. Flannery
                    // p.237
                    const T y = static_cast<T>(3.75) / arg;

                    const T p0 = static_cast<T>(0.392377e-2);
                    const T p1 = static_cast<T>(-0.1647633e-1) + y * p0;
                    const T p2 = static_cast<T>(0.2635537e-1) + y * p1;
                    const T p3 = static_cast<T>(-0.2057706e-1) + y * p2;
                    const T p4 = static_cast<T>(0.916281e-2) + y * p3;
                    const T p5 = static_cast<T>(-0.157565e-2) + y * p4;
                    const T p6 = static_cast<T>(0.225319e-2) + y * p5;
                    const T p7 = static_cast<T>(0.1328592e-1) + y * p6;
                    const T p8 = static_cast<T>(0.39894228) + y * p7;

                    return arg - std::sqrt(arg) + std::log(p8);
                }
            }
        };

        struct BesselFn_1_0 {
            template <typename T>
            __host__ __device__ constexpr T operator()(const T& arg) const noexcept
            {
                if (arg < static_cast<T>(3.75)) {
                    T bessel_0_p6 = 0;
                    const T ratio = arg / static_cast<T>(3.75);
                    const T y = ratio * ratio;

                    {
                        const T p0 = static_cast<T>(0.45813e-2);
                        const T p1 = static_cast<T>(0.360768e-1) + y * p0;
                        const T p2 = static_cast<T>(0.2659732) + y * p1;
                        const T p3 = static_cast<T>(1.2067492) + y * p2;
                        const T p4 = static_cast<T>(3.0899424) + y * p3;
                        const T p5 = static_cast<T>(3.5156229) + y * p4;
                        bessel_0_p6 = static_cast<T>(1.0) + y * p5;
                    }

                    T bessel_1_p6 = 0;
                    {
                        const T p0 = static_cast<T>(0.32411e-3);
                        const T p1 = static_cast<T>(0.301532e-2) + y * p0;
                        const T p2 = static_cast<T>(0.2658733e-1) + y * p1;
                        const T p3 = static_cast<T>(0.15084934) + y * p2;
                        const T p4 = static_cast<T>(0.51498869) + y * p3;
                        const T p5 = static_cast<T>(0.87890594) + y * p4;
                        bessel_1_p6 = arg * (static_cast<T>(0.5) + y * p5);
                    }

                    return bessel_1_p6 / bessel_0_p6;
                } else {
                    // see Numerical Recipes in C - 2nd Edition
                    // by W. H. Press, S. A. Teukolsky, W. T. Vetterling, B. P. Flannery
                    // p.238
                    const T y = static_cast<T>(3.75) / arg;

                    T bessel_0_p8;
                    {
                        const T p0 = static_cast<T>(0.392377e-2);
                        const T p1 = static_cast<T>(-0.1647633e-1) + y * p0;
                        const T p2 = static_cast<T>(0.2635537e-1) + y * p1;
                        const T p3 = static_cast<T>(-0.2057706e-1) + y * p2;
                        const T p4 = static_cast<T>(0.916281e-2) + y * p3;
                        const T p5 = static_cast<T>(-0.157565e-2) + y * p4;
                        const T p6 = static_cast<T>(0.225319e-2) + y * p5;
                        const T p7 = static_cast<T>(0.1328592e-1) + y * p6;
                        bessel_0_p8 = static_cast<T>(0.39894228) + y * p7;
                    }

                    T bessel_1_p8;
                    {
                        const T p0 = static_cast<T>(0.420059e-2);
                        const T p1 = static_cast<T>(0.1787654e-1) - y * p0;
                        const T p2 = static_cast<T>(-0.2895312e-1) + y * p1;
                        const T p3 = static_cast<T>(0.2282967e-1) + y * p2;
                        const T p4 = static_cast<T>(-0.1031555e-1) + y * p3;
                        const T p5 = static_cast<T>(0.163801e-2) + y * p4;
                        const T p6 = static_cast<T>(-0.362018e-2) + y * p5;
                        const T p7 = static_cast<T>(-0.3988024e-1) + y * p6;
                        bessel_1_p8 = static_cast<T>(0.39894228) + y * p7;
                    }

                    return bessel_1_p8 / bessel_0_p8;
                }
            }
        };
    } // namespace detail

    static constexpr __device__ detail::BesselFn_log_0 bessel_log_0;
    static constexpr __device__ detail::BesselFn_1_0 bessel_1_0;
} // namespace elsa::fn
