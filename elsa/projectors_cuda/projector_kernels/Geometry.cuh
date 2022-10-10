#pragma once

#include <cuda_runtime.h>
#include "SharedArray.cuh"

/// calculates the point at a distance delta from the ray origin ro in direction rd
template <typename real_t, uint32_t dim, typename InArray, typename OutArray>
__device__ __forceinline__ void pointAt(const real_t* const __restrict__ ro, const InArray& rd,
                                        const real_t delta, OutArray& result)
{
#pragma unroll
    for (uint32_t i = 0; i < dim; i++)
        result[i] = delta * rd[i] + ro[i];
}

/// projects a point onto the bounding box by clipping (points inside the bounding box are
/// unaffected)
template <typename real_t, uint32_t dim, typename OutArray, typename InArray>
__device__ __forceinline__ void projectOntoBox(OutArray& point, const InArray& boxMax)
{
#pragma unroll
    for (uint32_t i = 0; i < dim; i++) {
        point[i] = point[i] < 0.0f ? 0.0f : point[i];
        point[i] = point[i] > boxMax[i] ? boxMax[i] : point[i];
    }
}

/// find intersection points of ray with AABB
template <typename real_t, uint32_t dim, typename Array>
__device__ __forceinline__ bool box_intersect(const real_t* const __restrict__ ro,
                                              const real_t* const __restrict__ rd,
                                              const Array& boxMax, real_t& tmin, real_t& tmax)
{
    real_t invDir = 1.0f / rd[0];

    real_t t1 = -ro[0] * invDir;
    real_t t2 = (boxMax[0] - ro[0]) * invDir;

    /**
     * fmin and fmax adhere to the IEEE standard, and return the non-NaN element if only a single
     * NaN is present
     */
    // tmin and tmax have to be picked for each specific direction without using fmin/fmax
    // (suppressing NaNs is bad in this case)
    tmin = invDir >= 0 ? t1 : t2;
    tmax = invDir >= 0 ? t2 : t1;

#pragma unroll
    for (int i = 1; i < dim; ++i) {
        invDir = 1.0f / rd[i];

        t1 = -ro[i] * invDir;
        t2 = (boxMax[i] - ro[i]) * invDir;

        tmin = fmax(tmin, invDir >= 0 ? t1 : t2);
        tmax = fmin(tmax, invDir >= 0 ? t2 : t1);
    }

    if (tmax == 0.0f && tmin == 0.0f)
        return false;

    return tmax >= fmax(tmin, 0.0f);
}

template <typename real_t, uint32_t dim, typename RdArray, typename Array>
__device__ __forceinline__ bool box_intersect(const real_t* const __restrict__ ro, RdArray rd,
                                              Array boxMax, real_t& tmin, real_t& tmax)
{
    real_t invDir = 1.0f / rd[0];

    real_t t1 = -ro[0] * invDir;
    real_t t2 = (boxMax[0] - ro[0]) * invDir;

    /**
     * fmin and fmax adhere to the IEEE standard, and return the non-NaN element if only a single
     * NaN is present
     */
    // tmin and tmax have to be picked for each specific direction without using fmin/fmax
    // (suppressing NaNs is bad in this case)
    tmin = invDir >= 0 ? t1 : t2;
    tmax = invDir >= 0 ? t2 : t1;

#pragma unroll
    for (int i = 1; i < dim; ++i) {
        invDir = 1.0f / rd[i];

        t1 = -ro[i] * invDir;
        t2 = (boxMax[i] - ro[i]) * invDir;

        tmin = fmax(tmin, invDir >= 0 ? t1 : t2);
        tmax = fmin(tmax, invDir >= 0 ? t2 : t1);
    }

    if (tmax == 0.0f && tmin == 0.0f)
        return false;

    return tmax >= fmax(tmin, 0.0f);
}
