#include "TraverseJosephsCUDA.cuh"

#include <type_traits>

constexpr uint32_t MAX_THREADS_PER_BLOCK =
    elsa::TraverseJosephsCUDA<float, 3>::MAX_THREADS_PER_BLOCK;

template <typename data_t, uint32_t size>
struct EasyAccessSharedArray {
    data_t* const __restrict__ _p;

    __device__ EasyAccessSharedArray(data_t* p) : _p{p + threadIdx.x} {}

    __device__ __forceinline__ const data_t& operator[](uint32_t index) const
    {
        return _p[index * MAX_THREADS_PER_BLOCK];
    }

    __device__ __forceinline__ data_t& operator[](uint32_t index)
    {
        return _p[index * MAX_THREADS_PER_BLOCK];
    }
};

template <typename real_t, uint32_t dim, typename Array,
          typename = std::enable_if_t<std::is_same<Array, EasyAccessSharedArray<real_t, dim>>::value
                                      || std::is_same<Array, real_t*>::value>>
__device__ __forceinline__ void gesqmv(const int8_t* const __restrict__ matrix,
                                       const real_t* const __restrict__ vector, Array result,
                                       const uint32_t matrixPitch)
{
    // initialize result vector
    real_t* columnPtr = (real_t*) matrix;
#pragma unroll
    for (uint32_t x = 0; x < dim; x++) {
        result[x] = columnPtr[x] * vector[0];
    }

// accumulate results for remaning columns
#pragma unroll
    for (uint32_t y = 1; y < dim; y++) {
        real_t* columnPtr = (real_t*) (matrix + matrixPitch * y);
#pragma unroll
        for (uint32_t x = 0; x < dim; x++) {
            result[x] += columnPtr[x] * vector[y];
        }
    }
}

/// determine reverse norm of vector of length 2 or 3 using device inbuilt functions
template <typename real_t, uint32_t dim>
__device__ __forceinline__ real_t rnorm(const EasyAccessSharedArray<real_t, dim>& vector)
{
    if (dim == 3)
        return rnorm3d(vector[0], vector[1], vector[2]);
    else if (dim == 2)
        return rhypot(vector[0], vector[1]);
    else {
        real_t acc = vector[0];
#pragma unroll
        for (uint32_t i = 1; i < dim; i++)
            acc += vector[i];

        return acc;
    }
}

/// normalizes a vector of length 2 or 3 using device inbuilt norm
template <typename real_t, uint32_t dim>
__device__ __forceinline__ void normalize(EasyAccessSharedArray<real_t, dim>& vector)
{
    real_t rn = rnorm<real_t, dim>(vector);

#pragma unroll
    for (uint32_t i = 0; i < dim; i++) {
        vector[i] *= rn;
    }
}

/// calculates the point at a distance delta from the ray origin ro in direction rd
template <typename real_t, uint32_t dim>
__device__ __forceinline__ void
    pointAt(const real_t* const __restrict__ ro, const EasyAccessSharedArray<real_t, dim>& rd,
            const real_t delta, EasyAccessSharedArray<real_t, dim>& result)
{
#pragma unroll
    for (uint32_t i = 0; i < dim; i++)
        result[i] = delta * rd[i] + ro[i];
}

/// projects a point onto the bounding box by clipping (points inside the bounding box are
/// unaffected)
template <typename real_t, uint32_t dim>
__device__ __forceinline__ void projectOntoBox(EasyAccessSharedArray<real_t, dim>& point,
                                               const EasyAccessSharedArray<real_t, dim>& boxMax)
{
#pragma unroll
    for (uint32_t i = 0; i < dim; i++) {
        point[i] = point[i] < 0.0f ? 0.0f : point[i];
        point[i] = point[i] > boxMax[i] ? boxMax[i] : point[i];
    }
}

/// determines the voxel that contains a point, if the point is on a border the voxel in the ray
/// direction is favored
template <typename real_t, uint32_t dim>
__device__ __forceinline__ bool closestVoxel(const EasyAccessSharedArray<real_t, dim>& point,
                                             const EasyAccessSharedArray<real_t, dim>& boxMax,
                                             EasyAccessSharedArray<uint32_t, dim>& voxelCoord,
                                             const EasyAccessSharedArray<real_t, dim>& rd)
{
#pragma unroll
    for (uint32_t i = 0; i < dim; i++) {
        // point has been projected onto box => point[i]>=0, can use uint32_t
        uint32_t fl = trunc(point[i]);
        // for Joseph's also consider rays running along the "left" boundary
        voxelCoord[i] = fl == point[i] && rd[i] <= 0.0f && point[i] != 0.0f ? fl - 1 : fl;
        if (voxelCoord[i] >= boxMax[i])
            return false;
    }
    return true;
}

/// initialize step sizes considering the ray direcion
template <typename real_t, uint32_t dim>
__device__ __forceinline__ void initDelta(const EasyAccessSharedArray<real_t, dim> rd,
                                          EasyAccessSharedArray<real_t, dim> delta)
{
#pragma unroll
    for (uint32_t i = 0; i < dim; i++) {
        real_t d = ((rd[i] > 0.0f) - (rd[i] < 0.0f)) / rd[i];
        delta[i] = rd[i] >= -__FLT_EPSILON__ && rd[i] <= __FLT_EPSILON__ ? __FLT_MAX__ : d;
    }
}

template <typename real_t, uint32_t dim>
__device__ __forceinline__ bool
    boxIntersect(const real_t* const __restrict__ ro, const EasyAccessSharedArray<real_t, dim>& rd,
                 const EasyAccessSharedArray<real_t, dim>& boxMax, real_t& tmin, real_t& tmax)
{
    real_t invDir = 1.0f / rd[0];

    real_t t1 = -ro[0] * invDir;
    real_t t2 = (boxMax[0] - ro[0]) * invDir;

    /**
     * fminf and fmaxf adhere to the IEEE standard, and return the non-NaN element if only a single
     * NaN is present
     */
    // tmin and tmax have to be picked for each specific direction without using fmin/fmax
    // (supressing NaNs is bad in this case)
    tmin = invDir >= 0 ? t1 : t2;
    tmax = invDir >= 0 ? t2 : t1;

#pragma unroll
    for (uint32_t i = 1; i < dim; ++i) {
        invDir = 1.0f / rd[i];

        t1 = -ro[i] * invDir;
        t2 = (boxMax[i] - ro[i]) * invDir;

        tmin = fmax(tmin, invDir >= 0 ? t1 : t2);
        tmax = fmin(tmax, invDir >= 0 ? t2 : t1);
    }

    if (tmax == 0.0f && tmin == 0.0f)
        return false;
    if (tmax >= fmax(tmin, 0.0f)) // hit
        return true;
    return false;
}

/// returns the index of the smallest element in an array
template <typename real_t, uint32_t dim>
__device__ __forceinline__ uint32_t minIndex(const EasyAccessSharedArray<real_t, dim>& array)
{
    uint32_t index = 0;
    real_t min = array[0];

#pragma unroll
    for (uint32_t i = 1; i < dim; i++) {
        bool cond = array[i] < min;
        index = cond ? i : index;
        min = cond ? array[i] : min;
    }

    return index;
}

/// return the index of the element with the maximum absolute value in array
template <typename real_t, uint32_t dim>
__device__ __forceinline__ uint32_t maxAbsIndex(const EasyAccessSharedArray<real_t, dim>& array)
{
    uint32_t index = 0;
    real_t max = fabs(array[0]);

#pragma unroll
    for (uint32_t i = 1; i < dim; i++) {
        bool cond = fabs(array[i]) > max;
        index = cond ? i : index;
        max = cond ? fabs(array[i]) : max;
    }

    return index;
}

template <typename real_t, uint32_t dim>
__device__ __forceinline__ void updateTraverse(EasyAccessSharedArray<real_t, dim>& p,
                                               const EasyAccessSharedArray<real_t, dim>& rd,
                                               const real_t dist)
{
#pragma unroll
    for (uint32_t i = 0; i < dim; i++)
        p[i] += rd[i] * dist;
}

/// convenience function for texture fetching
template <typename real_t, uint32_t dim>
__device__ __forceinline__ real_t tex(cudaTextureObject_t texObj,
                                      const EasyAccessSharedArray<elsa::real_t, dim> p)
{
    if (dim == 3)
        return tex3D<real_t>(texObj, p[0], p[1], p[2]);
    else
        return tex2D<real_t>(texObj, p[0], p[1]);
}

/// fetches double at position (x,y) from 2D texture
__device__ __forceinline__ double tex2Dd(cudaTextureObject_t texObj, const float x, const float y)
{
    uint2 rt = tex2D<uint2>(texObj, x, y);
    return __hiloint2double(rt.y, rt.x);
}

/// template specialization for double texture fetches
template <>
__device__ __forceinline__ double tex<double, 2>(cudaTextureObject_t texObj,
                                                 const EasyAccessSharedArray<elsa::real_t, 2> p)
{
    elsa::real_t x = p[0] - 0.5f;
    elsa::real_t y = p[1] - 0.5f;

    elsa::real_t i = floor(x);
    elsa::real_t j = floor(y);

    elsa::real_t a = x - i;
    elsa::real_t b = y - j;

    double T[2][2];
    T[0][0] = tex2Dd(texObj, i, j);
    T[1][0] = tex2Dd(texObj, i + 1, j);
    T[0][1] = tex2Dd(texObj, i, j + 1);
    T[1][1] = tex2Dd(texObj, i + 1, j + 1);

    return (1 - a) * (1 - b) * T[0][0] + a * (1 - b) * T[1][0] + (1 - a) * b * T[0][1]
           + a * b * T[1][1];
}

/// fetches double at position (x,y,z) from 3D texture
__device__ __forceinline__ double tex3Dd(cudaTextureObject_t texObj, const float x, const float y,
                                         const float z)
{
    uint2 rt = tex3D<uint2>(texObj, x, y, z);
    return __hiloint2double(rt.y, rt.x);
}

/// template specialization for double texture fetches
template <>
__device__ __forceinline__ double tex<double, 3>(cudaTextureObject_t texObj,
                                                 const EasyAccessSharedArray<elsa::real_t, 3> p)
{
    elsa::real_t x = p[0] - 0.5f;
    elsa::real_t y = p[1] - 0.5f;
    elsa::real_t z = p[2] - 0.5f;

    elsa::real_t i = floor(x);
    elsa::real_t j = floor(y);
    elsa::real_t k = floor(z);

    elsa::real_t a = x - i;
    elsa::real_t b = y - j;
    elsa::real_t c = z - k;

    double T[2][2][2];
    T[0][0][0] = tex3Dd(texObj, i, j, k);
    T[1][0][0] = tex3Dd(texObj, i + 1, j, k);
    T[0][1][0] = tex3Dd(texObj, i, j + 1, k);
    T[0][0][1] = tex3Dd(texObj, i, j, k + 1);
    T[1][1][0] = tex3Dd(texObj, i + 1, j + 1, k);
    T[1][0][1] = tex3Dd(texObj, i + 1, j, k + 1);
    T[0][1][1] = tex3Dd(texObj, i, j + 1, k + 1);
    T[1][1][1] = tex3Dd(texObj, i + 1, j + 1, k + 1);

    return (1 - a) * (1 - b) * (1 - c) * T[0][0][0] + a * (1 - b) * (1 - c) * T[1][0][0] +

           (1 - a) * b * (1 - c) * T[0][1][0] + a * b * (1 - c) * T[1][1][0] +

           (1 - a) * (1 - b) * c * T[0][0][1] + a * (1 - b) * c * T[1][0][1] +

           (1 - a) * b * c * T[0][1][1] + a * b * c * T[1][1][1];
}

template <typename data_t, uint32_t dim>
__global__ void __launch_bounds__(elsa::TraverseJosephsCUDA<data_t, dim>::MAX_THREADS_PER_BLOCK)
    traverseForwardKernel(
        cudaTextureObject_t volume, int8_t* const __restrict__ sinogram,
        const uint64_t sinogramPitch, const uint32_t sinogramOffsetX,
        const int8_t* const __restrict__ rayOrigins, const uint32_t originPitch,
        const int8_t* const __restrict__ projInv, const uint32_t projPitch,
        const typename elsa::TraverseJosephsCUDA<data_t, dim>::BoundingBox boundingBox)
{

    using real_t = elsa::real_t;

    const int8_t* const projInvPtr = projInv + blockIdx.x * projPitch * dim;

    const real_t* const rayOrigin = (real_t*) (rayOrigins + blockIdx.x * originPitch);

    const uint32_t xCoord = sinogramOffsetX + blockDim.x * blockIdx.z + threadIdx.x;

    data_t* sinogramPtr =
        ((data_t*) (sinogram + (blockIdx.x * gridDim.y + blockIdx.y) * sinogramPitch) + xCoord);

    *sinogramPtr = 0;

    // homogenous pixel coordinates
    real_t pixelCoord[dim];
    pixelCoord[0] = xCoord + 0.5f;
    pixelCoord[dim - 1] = 1.0f;
    if (dim == 3)
        pixelCoord[1] = blockIdx.y + 0.5f;

    __shared__ real_t currentPositionsShared[dim * MAX_THREADS_PER_BLOCK];
    __shared__ real_t rayDirectionsShared[dim * MAX_THREADS_PER_BLOCK];
    __shared__ real_t boxMaxsShared[dim * MAX_THREADS_PER_BLOCK];

    EasyAccessSharedArray<real_t, dim> boxMax{boxMaxsShared};
#pragma unroll
    for (uint32_t i = 0; i < dim; ++i)
        boxMax[i] = boundingBox[i];

    // compute ray direction
    EasyAccessSharedArray<real_t, dim> rd{rayDirectionsShared};
    gesqmv<real_t, dim>(projInvPtr, pixelCoord, rd, projPitch);

    // determine main direction
    const uint32_t idx = maxAbsIndex<real_t, dim>(rd);
    const real_t rdMax = abs(rd[idx]);

    real_t rn = rnorm<real_t, dim>(rd);

    real_t weight = rn / rdMax;

// normalize ray direction to have length 1/-1 in the main direction
#pragma unroll
    for (uint32_t i = 0; i < dim; ++i)
        rd[i] /= rdMax;

    // find volume intersections
    real_t tmin, tmax;
    if (!boxIntersect<real_t, dim>(rayOrigin, rd, boxMax, tmin, tmax))
        return;

    EasyAccessSharedArray<real_t, dim> currentPosition{currentPositionsShared};
    pointAt<real_t, dim>(rayOrigin, rd, tmin, currentPosition);

    // truncate as currentPosition is non-negative
    const real_t fl = trunc(currentPosition[idx]);
    // for Joseph's also consider rays running along the "left" boundary
    const real_t firstBoundary = fl == currentPosition[idx] && rd[idx] < 0.0f ? fl - 1.0f : fl;

    // find distance to next plane orthogonal to primary diretion
    const real_t nextBoundary = rd[idx] > 0.0f ? firstBoundary + 1.0f : firstBoundary;
    real_t minDelta = (nextBoundary - currentPosition[idx]) / rd[idx];

    real_t intersectionLength = tmax - tmin;
    // first plane intersection may lie outside the bounding box
    if (intersectionLength < minDelta) {
        // use midpoint of entire ray intersection as a constant integration value
        updateTraverse<real_t, dim>(currentPosition, rd, intersectionLength * 0.5f);

        *sinogramPtr = weight * intersectionLength * tex<data_t, dim>(volume, currentPosition);
        return;
    }

    /**
     * otherwise first plane intersection inside bounding box
     * add first line segment and move to first interior point
     */
    updateTraverse<real_t, dim>(currentPosition, rd, minDelta * 0.5f);
    data_t pixelValue = weight * minDelta * tex<data_t, dim>(volume, currentPosition);

    // from here on use tmin as an indication of the current position along the ray
    tmin += minDelta;

    // if next point isn't last
    if (tmax - tmin > 1.0f) {
        updateTraverse<real_t, dim>(currentPosition, rd, (minDelta + 1.0f) * 0.5f);
        tmin += 1.0f;
        pixelValue += weight * tex<data_t, dim>(volume, currentPosition);

        // while interior intersection points remain
        while (tmax - tmin > 1.0f) {
            updateTraverse<real_t, dim>(currentPosition, rd, 1.0f);
            tmin += 1.0f;
            pixelValue += weight * tex<data_t, dim>(volume, currentPosition);
        }
    }

    updateTraverse<real_t, dim>(currentPosition, rd, (tmax - tmin + 1.0f) * 0.5f);
    pixelValue += weight * (tmax - tmin) * tex<data_t, dim>(volume, currentPosition);

    *sinogramPtr = pixelValue;
}

/// fetches double at position x, layer layer from a 1D layered texture
__device__ __forceinline__ double tex1DLayeredd(cudaTextureObject_t texObj, const float x,
                                                const int layer)
{
    uint2 rt = tex1DLayered<uint2>(texObj, x, layer);
    return __hiloint2double(rt.y, rt.x);
}

/// template specialization for layered texture fetches
template <>
double tex1DLayered<double>(cudaTextureObject_t texObj, elsa::real_t x, const int layer)
{
    x = x - 0.5f;

    elsa::real_t i = floor(x);

    elsa::real_t a = x - i;

    double T[2];
    T[0] = tex1DLayeredd(texObj, i, layer);
    T[1] = tex1DLayeredd(texObj, i + 1, layer);

    return (1 - a) * T[0] + a * T[1];
}

/// fetches double at position (x,y), layer layer from a 2D layered texture
__device__ __forceinline__ double tex2DLayeredd(cudaTextureObject_t texObj, const float x,
                                                const float y, const int layer)
{
    uint2 rt = tex2DLayered<uint2>(texObj, x, y, layer);
    return __hiloint2double(rt.y, rt.x);
}

/// template specialization for layered texture fetches
template <>
double tex2DLayered<double>(cudaTextureObject_t texObj, elsa::real_t x, elsa::real_t y,
                            const int layer)
{
    x = x - 0.5f;
    y = y - 0.5f;

    elsa::real_t i = floor(x);
    elsa::real_t j = floor(y);

    elsa::real_t a = x - i;
    elsa::real_t b = y - j;

    double T[2][2];
    T[0][0] = tex2DLayeredd(texObj, i, j, layer);
    T[1][0] = tex2DLayeredd(texObj, i + 1, j, layer);
    T[0][1] = tex2DLayeredd(texObj, i, j + 1, layer);
    T[1][1] = tex2DLayeredd(texObj, i + 1, j + 1, layer);

    return (1 - a) * (1 - b) * T[0][0] + a * (1 - b) * T[1][0] + (1 - a) * b * T[0][1]
           + a * b * T[1][1];
}

// TODO: check if sorting can be used to make this even faster
template <typename data_t, uint32_t dim>
__global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK)
    traverseAdjointFastKernel(int8_t* const __restrict__ volume, const uint64_t volumePitch,
                              const uint32_t volumeOffsetX, const uint32_t steps,
                              cudaTextureObject_t sinogram,
                              const int8_t* const __restrict__ rayOrigins,
                              const uint32_t originPitch, const int8_t* const __restrict__ proj,
                              const uint32_t projPitch, const uint32_t numAngles)
{

    using real_t = elsa::real_t;

    int x = volumeOffsetX + threadIdx.x;
    int y = blockIdx.y;
    int z = blockIdx.x;

    real_t voxelCenter[dim];
    voxelCenter[0] = x + 0.5f;
    voxelCenter[1] = y + 0.5f;
    if (dim == 3)
        voxelCenter[dim - 1] = z + 0.5f;

    extern __shared__ real_t valuesShared[];

    EasyAccessSharedArray<real_t, 1> values{valuesShared};
    for (uint32_t step = 0; step < steps; ++step)
        values[step] = 0;

    for (uint i = 0; i < numAngles; i++) {
        const int8_t* const projPtr = proj + i * projPitch * dim;
        const real_t* const rayOrigin = (real_t*) (rayOrigins + i * originPitch);

        voxelCenter[0] = x + 0.5f;

        real_t rd[dim];
#pragma unroll
        for (uint j = 0; j < dim; j++)
            rd[j] = voxelCenter[j] - rayOrigin[j];

        // compute ray direction
        for (uint32_t step = 0; step < steps; ++step) {

            real_t pixelCoord[dim];
            gesqmv<real_t, dim>(projPtr, rd, pixelCoord, projPitch);

            // convert to homogenous coordinates
            pixelCoord[0] /= pixelCoord[dim - 1];

            if (dim == 3) {
                pixelCoord[1] /= pixelCoord[dim - 1];
                values[step] += tex2DLayered<data_t>(sinogram, pixelCoord[0], pixelCoord[1], i);
            } else {
                values[step] += tex1DLayered<data_t>(sinogram, pixelCoord[0], i);
            }

            voxelCenter[0] += blockDim.x;
            rd[0] += blockDim.x;
        }
    }

    for (uint32_t step = 0; step < steps; ++step) {
        int x = volumeOffsetX + step * blockDim.x + threadIdx.x;
        data_t& voxelRef = *(data_t*) (volume + x * sizeof(data_t) + y * volumePitch
                                       + z * volumePitch * gridDim.y);
        voxelRef = values[step];
    }
}

/*
 * atomicAdd() for doubles is only supported on devices of compute capability 6.0 or higher
 * implementation taken straight from the CUDA C programming guide:
 * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
 */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ __forceinline__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

/// backprojects the weighted sinogram value to a given pixel
template <typename data_t, uint dim>
__device__ __forceinline__ void
    backproject2(int8_t* const __restrict__ volume, const EasyAccessSharedArray<uint64_t, dim>& p,
                 EasyAccessSharedArray<uint32_t, dim>& voxelCoord,
                 const EasyAccessSharedArray<elsa::real_t, dim>& voxelCoordf,
                 const EasyAccessSharedArray<elsa::real_t, dim>& boxMax,
                 const EasyAccessSharedArray<elsa::real_t, dim>& frac, const data_t weightedVal)
{

    data_t* volumeXPtr = (data_t*) (volume + p[0] * voxelCoord[0] + p[1] * voxelCoord[1]);
    data_t val = (1.0f - frac[1]) * weightedVal;
    atomicAdd(volumeXPtr, val);

    // volume[i,j+1]
    voxelCoord[1] = voxelCoord[1] < boxMax[1] - 1 ? voxelCoordf[1] + 1 : boxMax[1] - 1;
    volumeXPtr = (data_t*) (volume + p[0] * voxelCoord[0] + p[1] * voxelCoord[1]);
    val = frac[1] * weightedVal;
    atomicAdd(volumeXPtr, val);
}

/// backprojects the weighted sinogram value to a given voxel
template <typename data_t, uint dim>
__device__ __forceinline__ void
    backproject4(int8_t* const __restrict__ volume, const EasyAccessSharedArray<uint64_t, dim>& p,
                 EasyAccessSharedArray<uint32_t, dim>& voxelCoord,
                 const EasyAccessSharedArray<elsa::real_t, dim>& voxelCoordf,
                 const EasyAccessSharedArray<elsa::real_t, dim>& boxMax,
                 const EasyAccessSharedArray<elsa::real_t, dim>& frac, const data_t weightedVal)
{
    data_t* volumeXPtr =
        (data_t*) (volume + p[0] * voxelCoord[0] + p[1] * voxelCoord[1] + p[2] * voxelCoord[2]);
    data_t val = (1.0f - frac[1]) * (1.0f - frac[2]) * weightedVal;
    atomicAdd(volumeXPtr, val);
    // frac[0] is 0

    // volume[i,j+1,k]
    voxelCoord[1] = voxelCoord[1] < boxMax[1] - 1.0f ? voxelCoordf[1] + 1.0f : boxMax[1] - 1.0f;
    volumeXPtr =
        (data_t*) (volume + p[0] * voxelCoord[0] + p[1] * voxelCoord[1] + p[2] * voxelCoord[2]);
    val = frac[1] * (1.0f - frac[2]) * weightedVal;
    atomicAdd(volumeXPtr, val);

    // volume[i,j+1,k+1]
    voxelCoord[2] = voxelCoord[2] < boxMax[2] - 1.0f ? voxelCoordf[2] + 1.0f : boxMax[2] - 1.0f;
    volumeXPtr =
        (data_t*) (volume + p[0] * voxelCoord[0] + p[1] * voxelCoord[1] + p[2] * voxelCoord[2]);
    val = frac[1] * frac[2] * weightedVal;
    atomicAdd(volumeXPtr, val);

    // volume[i,j,k+1]
    voxelCoord[1] = voxelCoordf[1] < 0.0f ? 0 : voxelCoordf[1];
    volumeXPtr =
        (data_t*) (volume + p[0] * voxelCoord[0] + p[1] * voxelCoord[1] + p[2] * voxelCoord[2]);
    val = (1.0f - frac[1]) * frac[2] * weightedVal;
    atomicAdd(volumeXPtr, val);
}

/// convenience method for backprojecting to a given pixel/voxel
template <typename data_t, uint dim>
__device__ __forceinline__ void
    backproject(int8_t* const __restrict__ volume, const EasyAccessSharedArray<uint64_t, dim>& p,
                EasyAccessSharedArray<uint32_t, dim>& voxelCoord,
                const EasyAccessSharedArray<elsa::real_t, dim>& voxelCoordf,
                const EasyAccessSharedArray<elsa::real_t, dim>& boxMax,
                const EasyAccessSharedArray<elsa::real_t, dim>& frac, const data_t weightedVal)
{
    if (dim == 3)
        backproject4<data_t, dim>(volume, p, voxelCoord, voxelCoordf, boxMax, frac, weightedVal);
    else
        backproject2<data_t, dim>(volume, p, voxelCoord, voxelCoordf, boxMax, frac, weightedVal);
}

/// swaps the values of a and b
template <typename T>
__device__ __forceinline__ void swap(T& a, T& b)
{
    T c = a;
    a = b;
    b = c;
}

template <typename data_t, uint dim>
__global__ void __launch_bounds__(elsa::TraverseJosephsCUDA<data_t, dim>::MAX_THREADS_PER_BLOCK)
    traverseAdjointKernel(int8_t* const __restrict__ volume, const uint64_t volumePitch,
                          const int8_t* const __restrict__ sinogram, const uint64_t sinogramPitch,
                          const uint32_t sinogramOffsetX,
                          const int8_t* const __restrict__ rayOrigins, const uint32_t originPitch,
                          const int8_t* const __restrict__ projInv, const uint32_t projPitch,
                          typename elsa::TraverseJosephsCUDA<data_t, dim>::BoundingBox boundingBox)
{

    using real_t = elsa::real_t;

    const int8_t* const projInvPtr = projInv + blockIdx.x * projPitch * dim;

    const real_t* const rayOrigin = (real_t*) (rayOrigins + blockIdx.x * originPitch);

    const uint32_t xCoord = sinogramOffsetX + blockDim.x * blockIdx.z + threadIdx.x;

    data_t& sinogramVal =
        *((data_t*) (sinogram + (blockIdx.x * gridDim.y + blockIdx.y) * sinogramPitch) + xCoord);

    // homogenous pixel coordinates
    real_t pixelCoord[dim];
    pixelCoord[0] = xCoord + 0.5f;
    pixelCoord[dim - 1] = 1.0f;
    if (dim == 3)
        pixelCoord[1] = blockIdx.y + 0.5f;

    __shared__ real_t currentPositionsShared[dim * MAX_THREADS_PER_BLOCK];
    __shared__ real_t rayDirectionsShared[dim * MAX_THREADS_PER_BLOCK];
    __shared__ uint32_t voxelCoordsShared[dim * MAX_THREADS_PER_BLOCK];
    __shared__ real_t voxelCoordfsShared[dim * MAX_THREADS_PER_BLOCK];
    __shared__ real_t fracsShared[dim * MAX_THREADS_PER_BLOCK];
    __shared__ real_t tdeltasShared[dim * MAX_THREADS_PER_BLOCK];
    __shared__ uint64_t permutationsShared[dim * MAX_THREADS_PER_BLOCK];
    __shared__ real_t boxMaxsShared[dim * MAX_THREADS_PER_BLOCK];

    EasyAccessSharedArray<real_t, dim> boxMax{boxMaxsShared};
#pragma unroll
    for (uint32_t i = 0; i < dim; ++i)
        boxMax[i] = boundingBox[i];

    // compute ray direction
    EasyAccessSharedArray<real_t, dim> rd{rayDirectionsShared};
    gesqmv<real_t, dim>(projInvPtr, pixelCoord, rd, projPitch);
    normalize<real_t, dim>(rd);

    // find volume intersections
    real_t tmin, tmax;
    if (!boxIntersect<real_t, dim>(rayOrigin, rd, boxMax, tmin, tmax))
        return;

    EasyAccessSharedArray<real_t, dim> tdelta{tdeltasShared};
    initDelta<real_t, dim>(rd, tdelta);

    EasyAccessSharedArray<real_t, dim> currentPosition{currentPositionsShared};
    pointAt<real_t, dim>(rayOrigin, rd, tmin, currentPosition);
    projectOntoBox<real_t, dim>(currentPosition, boxMax);

    EasyAccessSharedArray<uint32_t, dim> voxelCoord{voxelCoordsShared};
    if (!closestVoxel<real_t, dim>(currentPosition, boxMax, voxelCoord, rd))
        return;

    // determine primary direction
    uint32_t idx = minIndex<real_t, dim>(tdelta);
    const int s = ((rd[idx] > 0.0f) - (rd[idx] < 0.0f));

    EasyAccessSharedArray<uint64_t, dim> permutation{permutationsShared};
    permutation[0] = sizeof(data_t);
    permutation[1] = volumePitch;
    if (dim == 3)
        permutation[dim - 1] = volumePitch * boxMax[1];

    // find distance to next plane orthogonal to primary diretion
    real_t nextBoundary = rd[idx] > 0.0f ? voxelCoord[idx] + 1 : voxelCoord[idx];
    real_t minDelta = (nextBoundary - currentPosition[idx]) / rd[idx];

    uint32_t entryDir = 0;
    for (uint i = 1; i < dim; i++)
        // current position is already projected onto the box, so a direct comparison works
        if (currentPosition[i] == 0 || currentPosition[i] == boxMax[i])
            entryDir = i;

    real_t intersectionLength = tmax - tmin;

    EasyAccessSharedArray<real_t, dim> voxelCoordf{voxelCoordfsShared};
    EasyAccessSharedArray<real_t, dim> frac{fracsShared};

    // subtract 0.5 from current position to get voxel coordinates
    for (uint i = 0; i < dim; i++) {
        currentPosition[i] -= 0.5f;
    }

    // permute indices, so that entry direction is at first index
    swap<real_t>(rd[0], rd[entryDir]);
    swap<real_t>(currentPosition[0], currentPosition[entryDir]);
    swap<uint32_t>(voxelCoord[0], voxelCoord[entryDir]);
    swap<real_t>(tdelta[0], tdelta[entryDir]);
    swap<real_t>(boxMax[0], boxMax[entryDir]);
    swap<uint64_t>(permutation[0], permutation[entryDir]);

    // first plane intersection may lie outside the bounding box
    if (intersectionLength < minDelta) {
        // use midpoint of entire ray intersection with bounding box as a constant integration value
        updateTraverse<real_t, dim>(currentPosition, rd, intersectionLength * 0.5f);
        for (uint i = 0; i < dim; i++) {
            voxelCoordf[i] = floorf(currentPosition[i]);
            frac[i] = currentPosition[i] - voxelCoordf[i];
            voxelCoord[i] = fmax(voxelCoordf[i], static_cast<real_t>(0));
        }
        backproject<data_t, dim>(volume, permutation, voxelCoord, voxelCoordf, boxMax, frac,
                                 intersectionLength * sinogramVal);
        return;
    }

    /**
     * otherwise first plane intersection inside bounding box
     * add first line segment and move to first interior point
     */
    updateTraverse<real_t, dim>(currentPosition, rd, minDelta * 0.5f);
    for (uint i = 0; i < dim; i++) {
        voxelCoordf[i] = floorf(currentPosition[i]);
        frac[i] = currentPosition[i] - voxelCoordf[i];
        voxelCoord[i] = fmax(voxelCoordf[i], static_cast<real_t>(0));
    }
    backproject<data_t, dim>(volume, permutation, voxelCoord, voxelCoordf, boxMax, frac,
                             minDelta * sinogramVal);
    // from here on use tmin as an indication of the current position along the ray
    tmin += minDelta;

    idx = minIndex<real_t, dim>(tdelta);
    // permute indices, so that primary direction is at first index
    swap<real_t>(rd[0], rd[idx]);
    swap<real_t>(currentPosition[0], currentPosition[idx]);
    swap<uint32_t>(voxelCoord[0], voxelCoord[idx]);
    swap<real_t>(tdelta[0], tdelta[idx]);
    swap<real_t>(boxMax[0], boxMax[idx]);
    swap<uint64_t>(permutation[0], permutation[idx]);

    // if next point isn't last
    if (tmax - tmin > tdelta[0]) {
        updateTraverse<real_t, dim>(currentPosition, rd, (minDelta + tdelta[0]) * 0.5f);
        minDelta = tdelta[0];
        tmin += minDelta;

        // set up values at idx manually, might lead to errors else
        currentPosition[0] = round(currentPosition[0]);
        frac[0] = 0.0f;
        voxelCoord[0] = (uint32_t) currentPosition[0];

        for (uint i = 1; i < dim; i++) {
            voxelCoordf[i] = floorf(currentPosition[i]);
            frac[i] = currentPosition[i] - voxelCoordf[i];
            voxelCoord[i] = fmax(static_cast<real_t>(0), voxelCoordf[i]);
        }
        backproject<data_t, dim>(volume, permutation, voxelCoord, voxelCoordf, boxMax, frac,
                                 minDelta * sinogramVal);

        // while interior intersection points remain
        while (tmin + minDelta < tmax) {
            updateTraverse<real_t, dim>(currentPosition, rd, minDelta);
            tmin += minDelta;

            voxelCoord[0] += s;
            for (uint i = 1; i < dim; i++) {
                voxelCoordf[i] = floorf(currentPosition[i]);
                frac[i] = currentPosition[i] - voxelCoordf[i];
                voxelCoord[i] = fmax(voxelCoordf[i], static_cast<real_t>(0));
            }
            backproject<data_t, dim>(volume, permutation, voxelCoord, voxelCoordf, boxMax, frac,
                                     minDelta * sinogramVal);
        }
    }

    updateTraverse<real_t, dim>(currentPosition, rd, (tmax + minDelta - tmin) * 0.5f);
    for (uint32_t i = 1; i < dim; i++) {
        // for large volumes numerical errors sometimes cause currentPosition of the last voxel
        // to lie outside boxMax although ideally it should not even exceed boxMax-0.5; currently
        // solved by readjusting the coordinates if needed
        // TODO: try updating the traversal using pointAt() instead
        voxelCoordf[i] = floorf(currentPosition[i]);
        frac[i] = currentPosition[i] - voxelCoordf[i];
        voxelCoord[i] = fmax(voxelCoordf[i], static_cast<real_t>(0));

        if (voxelCoord[i] >= boxMax[i]) {
            voxelCoord[i] = boxMax[i] - 1.0f;
            frac[i] = 0.5f;
        }
    }

    real_t mainDirPos = currentPosition[0];
    for (uint32_t i = 0; i < dim; i++) {
        // move to exit point
        currentPosition[i] = currentPosition[i] + 0.5f + rd[i] * (tmax - tmin) * 0.5f;

        // distance to border
        currentPosition[i] = fabs(fmin(currentPosition[i], boxMax[i] - currentPosition[i]));
    }

    // find direction closest to border
    uint32_t exitDir = minIndex<real_t, dim>(currentPosition);

    if (exitDir == 0) {
        // again handle this case manually
        voxelCoordf[0] = (real_t) voxelCoord[0] + s;
        voxelCoord[0] =
            fmin(fmax(voxelCoordf[0], static_cast<real_t>(0)), boxMax[0] - static_cast<real_t>(1));
        frac[0] = mainDirPos - voxelCoordf[0];
    } else {
        voxelCoordf[0] = floorf(mainDirPos);
        frac[0] = mainDirPos - voxelCoordf[0];
        voxelCoord[0] = fmax(static_cast<real_t>(0), voxelCoordf[0]);
    }

    // permute indices, so that exit direction is at first index
    swap<uint32_t>(voxelCoord[0], voxelCoord[exitDir]);
    swap<real_t>(voxelCoordf[0], voxelCoordf[exitDir]);
    swap<real_t>(frac[0], frac[exitDir]);
    swap<real_t>(boxMax[0], boxMax[exitDir]);
    swap<uint64_t>(permutation[0], permutation[exitDir]);
    backproject<data_t, dim>(volume, permutation, voxelCoord, voxelCoordf, boxMax, frac,
                             (tmax - tmin) * sinogramVal);
}

namespace elsa
{

    template <typename data_t, uint32_t dim>
    void TraverseJosephsCUDA<data_t, dim>::traverseForward(
        dim3 sinogramDims, int threads, cudaTextureObject_t volume, int8_t* __restrict__ sinogram,
        uint64_t sinogramPitch, const int8_t* __restrict__ rayOrigins, uint32_t originPitch,
        const int8_t* __restrict__ projInv, uint32_t projPitch, const BoundingBox& boxMax)
    {
        uint32_t numImageBlocks = sinogramDims.z / threads;
        uint32_t remaining = sinogramDims.z % threads;
        uint32_t offset = numImageBlocks * threads;

        if (numImageBlocks > 0) {
            dim3 grid(sinogramDims.x, sinogramDims.y, numImageBlocks);
            traverseForwardKernel<data_t, dim><<<grid, threads>>>(volume, sinogram, sinogramPitch,
                                                                  0, rayOrigins, originPitch,
                                                                  projInv, projPitch, boxMax);
        }

        if (remaining > 0) {
            cudaStream_t remStream;

            if (cudaStreamCreate(&remStream) != cudaSuccess)
                throw std::logic_error(
                    "TraverseJosephsCUDA: Couldn't create stream for remaining images");

            dim3 grid(sinogramDims.x, sinogramDims.y, 1);
            traverseForwardKernel<data_t, dim><<<grid, remaining, 0, remStream>>>(
                volume, sinogram, sinogramPitch, offset, rayOrigins, originPitch, projInv,
                projPitch, boxMax);

            if (cudaStreamDestroy(remStream) != cudaSuccess)
                throw std::logic_error("TraverseJosephsCUDA: Couldn't destroy cudaStream object");
        }
    }

    template <typename data_t, uint32_t dim>
    void TraverseJosephsCUDA<data_t, dim>::traverseAdjoint(
        dim3 sinogramDims, int threads, int8_t* __restrict__ volume, uint64_t volumePitch,
        const int8_t* __restrict__ sinogram, uint64_t sinogramPitch,
        const int8_t* __restrict__ rayOrigins, uint32_t originPitch,
        const int8_t* __restrict__ projInv, uint32_t projPitch, const BoundingBox& boxMax)
    {
        uint32_t numImageBlocks = sinogramDims.z / threads;
        uint32_t remaining = sinogramDims.z % threads;
        uint32_t offset = numImageBlocks * threads;

        if (numImageBlocks > 0) {
            dim3 grid(sinogramDims.x, sinogramDims.y, numImageBlocks);
            traverseAdjointKernel<data_t, dim>
                <<<grid, threads>>>(volume, volumePitch, sinogram, sinogramPitch, 0, rayOrigins,
                                    originPitch, projInv, projPitch, boxMax);
        }

        if (remaining > 0) {
            cudaStream_t remStream;

            if (cudaStreamCreate(&remStream) != cudaSuccess)
                throw std::logic_error(
                    "TraverseJosephsCUDA: Couldn't create stream for remaining images");

            dim3 grid(sinogramDims.x, sinogramDims.y, 1);
            traverseAdjointKernel<data_t, dim><<<grid, remaining, 0, remStream>>>(
                volume, volumePitch, sinogram, sinogramPitch, offset, rayOrigins, originPitch,
                projInv, projPitch, boxMax);

            if (cudaStreamDestroy(remStream) != cudaSuccess)
                throw std::logic_error("TraverseJosephsCUDA: Couldn't destroy cudaStream object");
        }
    }

    template <typename data_t, uint32_t dim>
    void TraverseJosephsCUDA<data_t, dim>::traverseAdjointFast(
        dim3 volumeDims, int threads, int8_t* __restrict__ volume, uint64_t volumePitch,
        cudaTextureObject_t sinogram, const int8_t* __restrict__ rayOrigins, uint32_t originPitch,
        const int8_t* __restrict__ proj, uint32_t projPitch, uint32_t numAngles)
    {
        uint32_t numImageBlocks = volumeDims.z / threads;
        uint32_t remaining = volumeDims.z % threads;
        uint32_t offset = numImageBlocks * threads;

        if (numImageBlocks > 0) {
            dim3 grid(volumeDims.x, volumeDims.y, 1);
            traverseAdjointFastKernel<data_t, dim>
                <<<grid, threads, numImageBlocks * MAX_THREADS_PER_BLOCK * sizeof(data_t)>>>(
                    volume, volumePitch, 0, numImageBlocks, sinogram, rayOrigins, originPitch, proj,
                    projPitch, numAngles);
        }

        if (remaining > 0) {
            cudaStream_t remStream;

            if (cudaStreamCreate(&remStream) != cudaSuccess)
                throw std::logic_error(
                    "TraverseJosephsCUDA: Couldn't create stream for remaining images");

            dim3 grid(volumeDims.x, volumeDims.y, 1);
            traverseAdjointFastKernel<data_t, dim>
                <<<grid, threads, remaining * sizeof(data_t), remStream>>>(
                    volume, volumePitch, offset, 1, sinogram, rayOrigins, originPitch, proj,
                    projPitch, numAngles);

            if (cudaStreamDestroy(remStream) != cudaSuccess)
                throw std::logic_error("TraverseJosephsCUDA: Couldn't destroy cudaStream object");
        }
    }

    // template instantiations
    template struct TraverseJosephsCUDA<float, 2>;
    template struct TraverseJosephsCUDA<float, 3>;

    template struct TraverseJosephsCUDA<double, 2>;
    template struct TraverseJosephsCUDA<double, 3>;
} // namespace elsa