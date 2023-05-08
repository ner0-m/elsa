#pragma once

#include <cuda_runtime.h>
#include "Luts.hpp"

/**
 * @brief General lut lookup and interpolation
 */
template <typename data_t, uint32_t N = DEFAULT_LUT_SIZE>
__device__ __forceinline__ data_t lut_lerp(const data_t* const __restrict__ lut,
                                           const data_t distance, const data_t radius)
{
    const data_t index = abs(distance) / radius * static_cast<data_t>(N - 1);
    if (index < 0 || index > static_cast<data_t>(N - 1)) {
        return 0;
    }

    // Get the two closes indices
    size_t a = floor(index);
    size_t b = ceil(index);

    // Get the function values
    const data_t fa = lut[a];
    const data_t fb = lut[b];

    auto t = index - static_cast<data_t>(a);

    return t * fb + (1 - t) * fa;
}