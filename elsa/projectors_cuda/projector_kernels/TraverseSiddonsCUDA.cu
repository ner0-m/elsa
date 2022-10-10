#include "TraverseSiddonsCUDA.cuh"
#include "SharedArray.cuh"
#include "Matrix.cuh"
#include "Geometry.cuh"
#include "TraversalUtils.cuh"
#include "AtomicAdd.cuh"

constexpr uint32_t MAX_THREADS_PER_BLOCK =
    elsa::TraverseSiddonsCUDA<float, 3>::MAX_THREADS_PER_BLOCK;

template <typename data_t, uint32_t size>
using EasyAccessSharedArray =
    elsa::detail::EasyAccessSharedArray<data_t, size, MAX_THREADS_PER_BLOCK>;

/// determines the voxel that contains a point, if the point is on a border the voxel in the ray
/// direction is favored
template <typename real_t, uint32_t dim>
__device__ __forceinline__ bool closestVoxel(const real_t* const __restrict__ point,
                                             const EasyAccessSharedArray<uint32_t, dim>& boxMax,
                                             EasyAccessSharedArray<uint32_t, dim>& voxelCoord,
                                             const real_t* const __restrict__ rd,
                                             const EasyAccessSharedArray<int, dim>& stepDir)
{
#pragma unroll
    for (int i = 0; i < dim; i++) {
        // point has been projected onto box => point[i]>=0, can use uint32_t
        real_t fl = trunc(point[i]);
        voxelCoord[i] = fl == point[i] && rd[i] < 0.0f ? fl - 1.0f : fl;
        if (voxelCoord[i] >= boxMax[i])
            return false;
    }
    return true;
}

/// initializes stepDir with the sign of rd
template <typename real_t, uint32_t dim>
__device__ __forceinline__ void initStepDirection(const real_t* const __restrict__ rd,
                                                  EasyAccessSharedArray<int, dim>& stepDir)
{
#pragma unroll
    for (int i = 0; i < dim; i++)
        stepDir[i] = ((rd[i] > 0.0f) - (rd[i] < 0.0f));
}

/// initialize maximum step parameters considering the ray direction
template <typename real_t, uint32_t dim>
__device__ __forceinline__ void initMax(const real_t* const __restrict__ rd,
                                        const EasyAccessSharedArray<uint32_t, dim>& currentVoxel,
                                        const real_t* const __restrict__ point,
                                        EasyAccessSharedArray<real_t, dim>& tmax)
{
    real_t nextBoundary;
#pragma unroll
    for (int i = 0; i < dim; i++) {
        nextBoundary = rd[i] > 0.0f ? currentVoxel[i] + 1 : currentVoxel[i];
        tmax[i] = rd[i] >= -__FLT_EPSILON__ && rd[i] <= __FLT_EPSILON__
                      ? __FLT_MAX__
                      : (nextBoundary - point[i]) / rd[i];
    }
}

/// checks whether the voxel lies inside the AABB
template <typename real_t, uint32_t dim>
__device__ __forceinline__ bool
    isVoxelInVolume(const EasyAccessSharedArray<uint32_t, dim>& currentVoxel,
                    const EasyAccessSharedArray<uint32_t, dim>& boxMax, const uint32_t& index)
{
    return currentVoxel[index] < boxMax[index];
}

/// updates the traversal algorithm, after update the current position will be the exit point from
/// current voxel
template <typename real_t, uint32_t dim>
__device__ __forceinline__ real_t updateTraverse(EasyAccessSharedArray<uint32_t, dim>& currentVoxel,
                                                 const EasyAccessSharedArray<int, dim>& stepDir,
                                                 const EasyAccessSharedArray<real_t, dim>& tdelta,
                                                 EasyAccessSharedArray<real_t, dim>& tmax,
                                                 real_t& texit, uint32_t& index)
{
    real_t tentry = texit;

    index = minIndex<real_t, dim>(tmax);
    texit = tmax[index];

    currentVoxel[index] += stepDir[index];
    tmax[index] += tdelta[index];

    return texit - tentry;
}

template <typename data_t, bool adjoint, uint32_t dim>
__global__ void __launch_bounds__(elsa::TraverseSiddonsCUDA<data_t, dim>::MAX_THREADS_PER_BLOCK)
    traverseVolume(int8_t* const __restrict__ volume, const uint64_t volumePitch,
                   int8_t* const __restrict__ sinogram, const uint64_t sinogramPitch,
                   const uint32_t sinogramOffsetX, const int8_t* const __restrict__ rayOrigins,
                   const uint32_t originPitch, const int8_t* const __restrict__ projInv,
                   const uint32_t projPitch,
                   const typename elsa::TraverseSiddonsCUDA<data_t, dim>::BoundingBox boundingBox)
{
    using real_t = elsa::real_t;

    const int8_t* const projInvPtr = projInv + blockIdx.x * projPitch * dim;

    const real_t* const rayOrigin = (real_t*) (rayOrigins + blockIdx.x * originPitch);

    const uint32_t xCoord = sinogramOffsetX + blockDim.x * blockIdx.z + threadIdx.x;

    data_t* sinogramPtr =
        ((data_t*) (sinogram + (blockIdx.x * gridDim.y + blockIdx.y) * sinogramPitch) + xCoord);

    // homogenous pixel coordinates
    real_t pixelCoord[dim];
    pixelCoord[0] = xCoord + 0.5f;
    pixelCoord[dim - 1] = 1.0f;
    if (dim == 3)
        pixelCoord[1] = blockIdx.y + 0.5f;

    __shared__ uint32_t currentVoxelsShared[MAX_THREADS_PER_BLOCK * dim];
    __shared__ int stepDirsShared[MAX_THREADS_PER_BLOCK * dim];
    __shared__ real_t tdeltasShared[MAX_THREADS_PER_BLOCK * dim];
    __shared__ real_t tmaxsShared[MAX_THREADS_PER_BLOCK * dim];
    __shared__ uint32_t boxMaxsShared[MAX_THREADS_PER_BLOCK * dim];

    EasyAccessSharedArray<uint32_t, dim> boxMax{boxMaxsShared};
#pragma unroll
    for (uint32_t i = 0; i < dim; i++)
        boxMax[i] = boundingBox[i];

    // compute ray direction
    real_t rd[dim];
    gesqmv<real_t, dim>(projInvPtr, pixelCoord, rd, projPitch);
    normalize<real_t, dim>(rd);

    // find volume intersections
    real_t tmin, tmaxf;
    if (!box_intersect<real_t, dim>(rayOrigin, rd, boxMax, tmin, tmaxf))
        return;

    real_t entryPoint[dim];
    pointAt<real_t, dim>(rayOrigin, rd, tmin, entryPoint);
    projectOntoBox<real_t, dim>(entryPoint, boxMax);

    EasyAccessSharedArray<int, dim> stepDir{stepDirsShared};
    initStepDirection<real_t, dim>(rd, stepDir);

    EasyAccessSharedArray<uint32_t, dim> currentVoxel{currentVoxelsShared};
    if (!closestVoxel<real_t, dim>(entryPoint, boxMax, currentVoxel, rd, stepDir))
        return;

    EasyAccessSharedArray<real_t, dim> tdelta{tdeltasShared};
    EasyAccessSharedArray<real_t, dim> tmax{tmaxsShared};
    initDelta<real_t, dim>(rd, stepDir, tdelta);
    initMax<real_t, dim>(rd, currentVoxel, entryPoint, tmax);

    uint32_t index;
    real_t texit = 0.0f;
    real_t pixelValue = 0.0f;

    data_t* volumeXPtr =
        dim == 3 ? (data_t*) (volume
                              + (boundingBox[1] * currentVoxel[2] + currentVoxel[1]) * volumePitch)
                       + currentVoxel[0]
                 : (data_t*) (volume + currentVoxel[1] * volumePitch) + currentVoxel[0];
    do {
        real_t d = updateTraverse<real_t, dim>(currentVoxel, stepDir, tdelta, tmax, texit, index);
        if (adjoint)
            atomicAdd(volumeXPtr, *sinogramPtr * d);
        else
            pixelValue += d * (*volumeXPtr);

        volumeXPtr =
            dim == 3
                ? (data_t*) (volume
                             + (boundingBox[1] * currentVoxel[2] + currentVoxel[1]) * volumePitch)
                      + currentVoxel[0]
                : (data_t*) (volume + currentVoxel[1] * volumePitch) + currentVoxel[0];
    } while (isVoxelInVolume<real_t, dim>(currentVoxel, boxMax, index));

    if (!adjoint)
        *sinogramPtr = pixelValue;
}

namespace elsa
{
    template <typename data_t, uint32_t dim>
    void TraverseSiddonsCUDA<data_t, dim>::traverseForward(
        const dim3 sinogramDims, const int threads, int8_t* const __restrict__ volume,
        const uint64_t volumePitch, int8_t* const __restrict__ sinogram,
        const uint64_t sinogramPitch, const int8_t* const __restrict__ rayOrigins,
        const uint32_t originPitch, const int8_t* const __restrict__ projInv,
        const uint32_t projPitch, const BoundingBox& boundingBox)
    {
        uint32_t numImgBlocks = sinogramDims.z / threads;
        uint32_t remaining = sinogramDims.z % threads;
        uint32_t offset = numImgBlocks * threads;

        if (numImgBlocks > 0) {
            const dim3 grid(sinogramDims.x, sinogramDims.y, numImgBlocks);
            traverseVolume<data_t, false, dim>
                <<<grid, threads>>>(volume, volumePitch, sinogram, sinogramPitch, 0, rayOrigins,
                                    originPitch, projInv, projPitch, boundingBox);
        }

        if (remaining > 0) {
            cudaStream_t remStream;
            if (cudaStreamCreate(&remStream) != cudaSuccess)
                throw std::logic_error(
                    "TraverseSiddonsCUDA: Couldn't create stream for remaining images");

            const dim3 grid(sinogramDims.x, sinogramDims.y, 1);
            traverseVolume<data_t, false, dim><<<grid, remaining, 0, remStream>>>(
                volume, volumePitch, sinogram, sinogramPitch, offset, rayOrigins, originPitch,
                projInv, projPitch, boundingBox);

            if (cudaStreamDestroy(remStream) != cudaSuccess)
                throw std::logic_error("TraverseSiddonsCUDA: Couldn't destroy GPU stream; This may "
                                       "cause problems later.");
        }
    }

    template <typename data_t, uint32_t dim>
    void TraverseSiddonsCUDA<data_t, dim>::traverseAdjoint(
        const dim3 sinogramDims, const int threads, int8_t* const __restrict__ volume,
        const uint64_t volumePitch, int8_t* const __restrict__ sinogram,
        const uint64_t sinogramPitch, const int8_t* const __restrict__ rayOrigins,
        const uint32_t originPitch, const int8_t* const __restrict__ projInv,
        const uint32_t projPitch, const BoundingBox& boundingBox)
    {
        uint32_t numImgBlocks = sinogramDims.z / threads;
        uint32_t remaining = sinogramDims.z % threads;
        uint32_t offset = numImgBlocks * threads;

        if (numImgBlocks > 0) {
            const dim3 grid(sinogramDims.x, sinogramDims.y, numImgBlocks);
            traverseVolume<data_t, true, dim>
                <<<grid, threads>>>(volume, volumePitch, sinogram, sinogramPitch, 0, rayOrigins,
                                    originPitch, projInv, projPitch, boundingBox);
        }

        if (remaining > 0) {
            cudaStream_t remStream;
            if (cudaStreamCreate(&remStream) != cudaSuccess)
                throw std::logic_error(
                    "TraverseSiddonsCUDA: Couldn't create stream for remaining images");

            const dim3 grid(sinogramDims.x, sinogramDims.y, 1);
            traverseVolume<data_t, true, dim><<<grid, remaining, 0, remStream>>>(
                volume, volumePitch, sinogram, sinogramPitch, offset, rayOrigins, originPitch,
                projInv, projPitch, boundingBox);

            if (cudaStreamDestroy(remStream) != cudaSuccess)
                throw std::logic_error("TraverseSiddonsCUDA: Couldn't destroy GPU stream; This may "
                                       "cause problems later.");
        }
    }

    // ------------------------------------------
    // explicit template instantiation
    template struct TraverseSiddonsCUDA<float, 2>;
    template struct TraverseSiddonsCUDA<float, 3>;
    template struct TraverseSiddonsCUDA<double, 2>;
    template struct TraverseSiddonsCUDA<double, 3>;
} // namespace elsa
