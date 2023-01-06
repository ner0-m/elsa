#include "ProjectVoxelsCUDA.cuh"
#include "SharedArray.cuh"
#include "Matrix.cuh"
#include "Geometry.cuh"
#include "Lut.cuh"
#include "TraversalUtils.cuh"
#include "AtomicAdd.cuh"
#include "Math.hpp"

constexpr uint32_t MAX_THREADS_PER_BLOCK = elsa::ProjectVoxelsCUDA<float, 3>::MAX_THREADS_PER_BLOCK;

template <typename data_t, uint32_t size>
using EasyAccessSharedArray =
    elsa::detail::EasyAccessSharedArray<data_t, size, MAX_THREADS_PER_BLOCK>;
template <typename T>
__device__ __forceinline__ int sgn(T val)
{
    return (T(0) < val) - (val < T(0));
}

template <typename data_t, bool adjoint, elsa::VoxelHelperCUDA::PROJECTOR_TYPE type>
__global__ void __launch_bounds__(elsa::ProjectVoxelsCUDA<data_t, 2>::MAX_THREADS_PER_BLOCK)
    traverseVolume2D(data_t* const __restrict__ volume, const dim3 volumeDims,
                     data_t* const __restrict__ sinogram, const dim3 sinogramDims,
                     const uint32_t volumeOffsetX, const data_t* const __restrict__ proj,
                     const data_t* const __restrict__ ext, const data_t* __restrict__ lut,
                     const data_t radius, const data_t sourceDetectorDistance)
{
    using index_t = elsa::index_t;

    uint32_t geomIdx = blockIdx.z;

    data_t* detectorZeroIndex = sinogram + geomIdx * sinogramDims.x * sinogramDims.y;

    const data_t* const projPtr = proj + geomIdx * 2 * (2 + 1);
    const data_t* const extPtr = ext + geomIdx * 2 * (2 + 1);

    // homogenous pixel coordinates
    data_t vCoord[2 + 1];
    vCoord[0] = volumeOffsetX + threadIdx.x + blockIdx.x * blockDim.x;
    vCoord[1] = blockIdx.y;
    vCoord[2] = 1.0f;

    unsigned int volumeIndex = vCoord[0] + vCoord[1] * volumeDims.x;

    // shift to center of voxel
    vCoord[0] += 0.5;
    vCoord[1] += 0.5;

    // compute voxel on detector
    data_t dCoord[2];
    gehomv<2, data_t>(projPtr, vCoord, dCoord);
    homogenousNormalize<data_t, 2>(dCoord);

    // correct shift
    dCoord[0] -= 0.5f;

    // compute voxel in camera space for length
    data_t cCoord[2];
    gehomv<2, data_t>(extPtr, vCoord, cCoord);
    data_t cDistance = hypot(cCoord[0], cCoord[1]);
    data_t scaling = abs(sourceDetectorDistance / cDistance);
    auto radiusOnDetector = radius * scaling;

    // find all detector pixels that are hit
    auto lower = max((index_t) 0, static_cast<index_t>(ceil(dCoord[0] - radiusOnDetector)));
    auto upper = min((index_t) sinogramDims.x - 1,
                     static_cast<index_t>(floor(dCoord[0] + radiusOnDetector)));
    for (index_t neighbour = lower; neighbour <= upper; neighbour++) {
        data_t weight;
        const data_t distance = dCoord[0] - neighbour;

        if constexpr (type == elsa::VoxelHelperCUDA::CLASSIC)
            weight = lut_lerp<data_t, 100>(lut, distance / scaling, radius);
        else if constexpr (type == elsa::VoxelHelperCUDA::DIFFERENTIAL)
            weight = sgn(distance) * lut_lerp<data_t, 100>(lut, distance / scaling, radius);

        if constexpr (adjoint) {
            atomicAdd(volume + volumeIndex, weight * *(detectorZeroIndex + neighbour));
        } else {
            atomicAdd(detectorZeroIndex + neighbour, weight * volume[volumeIndex]);
        }
    }
}

template <typename data_t, bool adjoint, elsa::VoxelHelperCUDA::PROJECTOR_TYPE type>
__global__ void __launch_bounds__(elsa::ProjectVoxelsCUDA<data_t, 3>::MAX_THREADS_PER_BLOCK)
    traverseVolume3D(data_t* const __restrict__ volume, const dim3 volumeDims,
                     data_t* const __restrict__ sinogram, const dim3 sinogramDims,
                     const uint32_t volumeOffsetX, uint32_t geomIdx,
                     const data_t* const __restrict__ proj, const data_t* const __restrict__ ext,
                     const data_t* __restrict__ lut, const data_t radius,
                     const data_t sourceDetectorDistance)
{
    using index_t = elsa::index_t;

    data_t* detectorZeroIndex = sinogram + geomIdx * sinogramDims.x * sinogramDims.y;

    const data_t* const projPtr = proj + geomIdx * 3 * (3 + 1);
    const data_t* const extPtr = ext + geomIdx * 3 * (3 + 1);

    // homogenous pixel coordinates
    data_t vCoord[3 + 1];
    vCoord[0] = volumeOffsetX + threadIdx.x + blockIdx.x * blockDim.x;
    vCoord[1] = blockIdx.y;
    vCoord[2] = blockIdx.z;
    vCoord[3] = 1.0f;

    unsigned int volumeIndex =
        vCoord[0] + vCoord[1] * volumeDims.x + volumeDims.y * volumeDims.x * vCoord[2];

    // shift to center of voxel
    vCoord[0] += 0.5;
    vCoord[1] += 0.5;
    vCoord[2] += 0.5;

    // compute voxel on detector
    data_t dCoord[3];
    gehomv<3, data_t>(projPtr, vCoord, dCoord);
    homogenousNormalize<data_t, 3>(dCoord);

    // correct shift
    dCoord[0] -= 0.5f;
    dCoord[1] -= 0.5f;

    // compute voxel in camera space for length
    data_t cCoord[3];
    gehomv<3, data_t>(extPtr, vCoord, cCoord);
    data_t cDistance = norm3d(cCoord[0], cCoord[1], cCoord[2]);
    data_t scaling = abs(sourceDetectorDistance / cDistance);
    auto radiusOnDetector = radius * scaling;

    // find all detector pixels that are hit
    // compute bounding box of hit detector Pixels
    index_t lowerIndex[2];
    index_t upperIndex[2];

    lowerIndex[0] = max((index_t) 0, static_cast<index_t>(ceil(dCoord[0] - radiusOnDetector)));
    lowerIndex[1] = max((index_t) 0, static_cast<index_t>(ceil(dCoord[1] - radiusOnDetector)));
    upperIndex[0] = min((index_t) sinogramDims.x - 1,
                        static_cast<index_t>(floor(dCoord[0] + radiusOnDetector)));
    upperIndex[1] = min((index_t) sinogramDims.y - 1,
                        static_cast<index_t>(floor(dCoord[1] + radiusOnDetector)));

    // initialize variables for performance
    auto iStride = upperIndex[0] - lowerIndex[0] + 1;
    auto jStride = sinogramDims.x + 1 - iStride;
    data_t* curDetectorIdx = detectorZeroIndex + lowerIndex[0] + lowerIndex[1] * sinogramDims.x;
    data_t currentCoord[2];
    currentCoord[0] = lowerIndex[0];
    currentCoord[1] = lowerIndex[1];

    for (index_t j = lowerIndex[1]; j <= upperIndex[1]; j++) {
        for (index_t i = lowerIndex[0]; i <= upperIndex[0]; i++) {
            auto primDistance = currentCoord[0] - dCoord[0];
            const data_t distance = hypot(primDistance, currentCoord[1] - dCoord[1]);
            data_t weight;

            // classic just compute the weight
            if constexpr (type == elsa::VoxelHelperCUDA::CLASSIC)
                weight = lut_lerp<data_t, 100>(lut, distance / scaling, radius);
            // differential the formula is projected_blob_derivative(||x||) / ||x|| * x_prim
            // prim is the first dimension here and the lut already takes care of the / ||x||
            else if constexpr (type == elsa::VoxelHelperCUDA::DIFFERENTIAL)
                weight =
                    primDistance / scaling * lut_lerp<data_t, 100>(lut, distance / scaling, radius);

            if constexpr (adjoint) {
                atomicAdd(volume + volumeIndex, weight * *curDetectorIdx);
            } else {
                atomicAdd(curDetectorIdx, weight * volume[volumeIndex]);
            }
            curDetectorIdx += 1;
            currentCoord[0] += 1;
        }
        curDetectorIdx += jStride;
        currentCoord[0] -= iStride;
        currentCoord[1] += 1;
    }
}

namespace elsa
{
    template <typename data_t, uint32_t dim, bool adjoint, VoxelHelperCUDA::PROJECTOR_TYPE type>
    void ProjectVoxelsCUDA<data_t, dim, adjoint, type>::project(
        const dim3 volumeDims, const dim3 sinogramDims, const int threads,
        data_t* __restrict__ volume, data_t* __restrict__ sinogram, const data_t* __restrict__ proj,
        const data_t* __restrict__ ext, const data_t* __restrict__ lut, const data_t radius,
        const data_t sdd)
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
                traverseVolume2D<data_t, adjoint, type><<<grid, threads, 0, mainStream>>>(
                    volume, volumeDims, sinogram, sinogramDims, 0, proj, ext, lut, radius, sdd);
            } else {
                // use last index for z
                grid = dim3{xBlocks, volumeDims.y, volumeDims.z};
                for (unsigned int geomIdx = 0; geomIdx < sinogramDims.z; geomIdx++) {
                    traverseVolume3D<data_t, adjoint, type><<<grid, threads, 0, mainStream>>>(
                        volume, volumeDims, sinogram, sinogramDims, 0, geomIdx, proj, ext, lut,
                        radius, sdd);
                }
            }
        }

        if (xRemaining > 0) {
            dim3 grid;
            if constexpr (dim == 2) {
                // use last index for geometry
                grid = dim3{1, volumeDims.y, sinogramDims.z};
                traverseVolume2D<data_t, adjoint, type><<<grid, xRemaining, 0, mainStream>>>(
                    volume, volumeDims, sinogram, sinogramDims, xOffset, proj, ext, lut, radius,
                    sdd);
            } else {
                // use last index for z
                grid = dim3{1, volumeDims.y, volumeDims.z};
                for (unsigned int geomIdx = 0; geomIdx < sinogramDims.z; geomIdx++) {
                    traverseVolume3D<data_t, adjoint, type><<<grid, xRemaining, 0, mainStream>>>(
                        volume, volumeDims, sinogram, sinogramDims, xOffset, geomIdx, proj, ext,
                        lut, radius, sdd);
                }
            }
        }

        if (cudaStreamDestroy(mainStream) != cudaSuccess)
            throw std::logic_error("ProjectVoxelsCUDA: Couldn't destroy main GPU stream; This may "
                                   "cause problems later.");
        cudaDeviceSynchronize();
    }

    // ------------------------------------------
    // explicit template instantiation

    // CLASSIC
    // 2D
    template struct ProjectVoxelsCUDA<float, 2, true, VoxelHelperCUDA::CLASSIC>;
    template struct ProjectVoxelsCUDA<float, 2, false, VoxelHelperCUDA::CLASSIC>;
    template struct ProjectVoxelsCUDA<double, 2, true, VoxelHelperCUDA::CLASSIC>;
    template struct ProjectVoxelsCUDA<double, 2, false, VoxelHelperCUDA::CLASSIC>;
    // 3D
    template struct ProjectVoxelsCUDA<float, 3, true, VoxelHelperCUDA::CLASSIC>;
    template struct ProjectVoxelsCUDA<float, 3, false, VoxelHelperCUDA::CLASSIC>;
    template struct ProjectVoxelsCUDA<double, 3, true, VoxelHelperCUDA::CLASSIC>;
    template struct ProjectVoxelsCUDA<double, 3, false, VoxelHelperCUDA::CLASSIC>;

    // DIFFERENTIAL
    // 2D
    template struct ProjectVoxelsCUDA<float, 2, true, VoxelHelperCUDA::DIFFERENTIAL>;
    template struct ProjectVoxelsCUDA<float, 2, false, VoxelHelperCUDA::DIFFERENTIAL>;
    template struct ProjectVoxelsCUDA<double, 2, true, VoxelHelperCUDA::DIFFERENTIAL>;
    template struct ProjectVoxelsCUDA<double, 2, false, VoxelHelperCUDA::DIFFERENTIAL>;
    // 3D
    template struct ProjectVoxelsCUDA<float, 3, true, VoxelHelperCUDA::DIFFERENTIAL>;
    template struct ProjectVoxelsCUDA<float, 3, false, VoxelHelperCUDA::DIFFERENTIAL>;
    template struct ProjectVoxelsCUDA<double, 3, true, VoxelHelperCUDA::DIFFERENTIAL>;
    template struct ProjectVoxelsCUDA<double, 3, false, VoxelHelperCUDA::DIFFERENTIAL>;
} // namespace elsa
