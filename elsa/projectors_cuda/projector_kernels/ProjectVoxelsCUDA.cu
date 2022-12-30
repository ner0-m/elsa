#include "ProjectVoxelsCUDA.cuh"
#include "SharedArray.cuh"
#include "Matrix.cuh"
#include "Geometry.cuh"
#include "Lut.cuh"
#include "TraversalUtils.cuh"
#include "AtomicAdd.cuh"
#include <array>

constexpr uint32_t MAX_THREADS_PER_BLOCK = elsa::ProjectVoxelsCUDA<float, 3>::MAX_THREADS_PER_BLOCK;

template <typename data_t, uint32_t size>
using EasyAccessSharedArray =
    elsa::detail::EasyAccessSharedArray<data_t, size, MAX_THREADS_PER_BLOCK>;

template <typename data_t, bool adjoint, uint32_t dim>
__global__ void __launch_bounds__(elsa::ProjectVoxelsCUDA<data_t, dim>::MAX_THREADS_PER_BLOCK)
    traverseVolume(data_t* const __restrict__ volume, const dim3 volumeDims,
                   data_t* const __restrict__ sinogram, const dim3 sinogramDims,
                   const uint32_t volumeOffsetX, const uint32_t volumeOffsetY,
                   const int8_t* const __restrict__ proj, const uint32_t projPitch,
                   const int8_t* const __restrict__ ext, const uint32_t extPitch,
                   const data_t* __restrict__ lut, const data_t radius,
                   const elsa::real_t sourceDetectorDistance)
{
    using real_t = elsa::real_t;

    data_t* const detectorZeroIndex = sinogram + blockIdx.z * sinogramDims.x * sinogramDims.y;

    const int8_t* const projPtr = proj + blockIdx.z * projPitch * (dim + 1);
    const int8_t* const extPtr = ext + blockIdx.z * extPitch * (dim + 1);

    // homogenous pixel coordinates
    real_t vCoord[dim + 1];
    vCoord[0] = volumeOffsetX + threadIdx.x + blockIdx.x * blockDim.x;
    vCoord[1] = volumeOffsetY + blockIdx.y; // threadIdx.y + blockIdx.y * blockDim.y;
    vCoord[dim] = 1.0f;
    if constexpr (dim == 3)
        vCoord[2] = blockIdx.z;

    unsigned int volumeIndex;
    if constexpr (dim == 2)
        volumeIndex = vCoord[0] + vCoord[1] * volumeDims.x;
    else
        volumeIndex =
            vCoord[0] + vCoord[1] * volumeDims.x + volumeDims.y * volumeDims.x * vCoord[2];

    // shift to center of voxel
    vCoord[0] += 0.5;
    vCoord[1] += 0.5;
    if constexpr (dim == 3)
        vCoord[2] += 0.5;

    // compute voxel on detector
    real_t dCoord[dim];
    gehomv<real_t, dim>(projPtr, vCoord, dCoord, projPitch);
    homogenousNormalize<real_t, dim>(dCoord);

    // correct shift
    dCoord[0] -= 0.5f;
    if constexpr (dim == 3)
        dCoord[1] -= 0.5f;

    // compute voxel in camera space for length
    real_t cCoord[dim];
    gehomv<real_t, dim>(extPtr, vCoord, cCoord, extPitch);
    real_t cDistance = norm<real_t, dim>(cCoord);
    real_t scaling = sourceDetectorDistance / cDistance;

    // find all detector pixels that are hit
    if constexpr (dim == 2) {
        auto radiusOnDetector = static_cast<size_t>(round(radius * scaling));
        size_t detectorIndex = static_cast<size_t>(round(dCoord[0]));

        auto lower = max((size_t) 0, detectorIndex - radiusOnDetector);
        auto upper = min((size_t) sinogramDims.x - 1, detectorIndex + radiusOnDetector);
        for (size_t neighbour = lower; neighbour <= upper; neighbour++) {
            const data_t distance = dCoord[0] - neighbour;
            auto weight = lut_lerp<data_t, 100>(lut, abs(distance / scaling / radius) * 100);
            if constexpr (adjoint) {
                atomicAdd(volume + volumeIndex, weight * *(detectorZeroIndex + neighbour));
            } else {
                atomicAdd(detectorZeroIndex + neighbour, weight * volume[volumeIndex]);
            }
        }
    }
}

namespace elsa
{
    template <typename data_t, uint32_t dim, bool adjoint>
    void ProjectVoxelsCUDA<data_t, dim, adjoint>::project(
        const dim3 volumeDims, const dim3 sinogramDims, const int threads,
        data_t* __restrict__ volume, data_t* __restrict__ sinogram, const int8_t* __restrict__ proj,
        const uint32_t projPitch, const int8_t* __restrict__ ext, const uint32_t extPitch,
        const data_t* __restrict__ lut, const data_t radius, const real_t sdd)
    {
        uint32_t xBlocks = volumeDims.x / threads;
        uint32_t xRemaining = volumeDims.x % threads;
        uint32_t xOffset = xBlocks * threads;

        cudaStream_t mainStream;
        if (cudaStreamCreate(&mainStream) != cudaSuccess)
            throw std::logic_error("ProjectVoxelsCUDA: Couldn't create main stream");

        if (xBlocks > 0) {
            dim3 grid;
            if constexpr (dim == 2) {
                // use last index for geometry
                grid = dim3{xBlocks, volumeDims.y, sinogramDims.z};
            } else {
                // use last index for z
                grid = dim3{xBlocks, volumeDims.y, volumeDims.z};
            }
            traverseVolume<data_t, adjoint, dim><<<grid, threads, 0, mainStream>>>(
                volume, volumeDims, sinogram, sinogramDims, 0, 0, proj, projPitch, ext, extPitch,
                lut, radius, sdd);
        }

        if (xRemaining > 0) {
            xRemaining = std::max(1u, xRemaining);

            dim3 grid;
            if constexpr (dim == 2) {
                // use last index for geometry
                grid = dim3{1, volumeDims.y, sinogramDims.z};
            } else {
                // use last index for z
                grid = dim3{1, volumeDims.y, volumeDims.z};
            }
            traverseVolume<data_t, adjoint, dim><<<grid, xRemaining, 0, mainStream>>>(
                volume, volumeDims, sinogram, sinogramDims, xOffset, 0, proj, projPitch, ext,
                extPitch, lut, radius, sdd);
        }

        if (cudaStreamDestroy(mainStream) != cudaSuccess)
            throw std::logic_error("ProjectVoxelsCUDA: Couldn't destroy main GPU stream; This may "
                                   "cause problems later.");
        cudaDeviceSynchronize();
    }

    // ------------------------------------------
    // explicit template instantiation
    template struct ProjectVoxelsCUDA<float, 2, true>;
    template struct ProjectVoxelsCUDA<float, 3, true>;
    template struct ProjectVoxelsCUDA<double, 2, true>;
    template struct ProjectVoxelsCUDA<double, 3, true>;
    template struct ProjectVoxelsCUDA<float, 2, false>;
    template struct ProjectVoxelsCUDA<float, 3, false>;
    template struct ProjectVoxelsCUDA<double, 2, false>;
    template struct ProjectVoxelsCUDA<double, 3, false>;
} // namespace elsa
