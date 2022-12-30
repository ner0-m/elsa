#pragma once

#include <cuda_runtime.h>

/**
 * @brief General lut lookup and interpolation
 *
 * important: always use byte pointers for multidimensional arrays
 */
template <typename real_t, uint32_t N>
__device__ __forceinline__ real_t lut_lerp(const real_t* const __restrict__ lut, const real_t index)
{
    if (index < 0 || index > N) {
        return 0;
    }

    // Get the two closes indices
    size_t a = floor(index);
    size_t b = ceil(index);

    // Get the function values
    const real_t fa = lut[a];
    const real_t fb = lut[b];

    auto t = index - static_cast<real_t>(a);

    return t * fa + (1 - t) * fb;
}