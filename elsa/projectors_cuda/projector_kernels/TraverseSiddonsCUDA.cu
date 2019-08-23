#include "TraverseSiddonsCUDA.cuh"

/**
 * \brief General square matrix-vector multiplication
 *
 * important: always use byte pointers for multidimensional arrays
 */
template <typename real_t, uint32_t dim> 
__device__ __forceinline__ void gesqmv(const int8_t* const __restrict__ matrix,
                                       const real_t* const __restrict__ vector,
                                       real_t* const __restrict__ result,
                                       const uint32_t matrixPitch)
{
    //initialize result vector
    real_t* columnPtr = (real_t*)matrix;
    #pragma unroll
    for (uint32_t x=0; x<dim; x++) {
        result[x] = columnPtr[x]*vector[0];
    }

    //accumulate results for remaning columns
    #pragma unroll
    for (uint32_t y = 1; y<dim; y++) {
        real_t* columnPtr = (real_t*)(matrix + matrixPitch*y);
        #pragma unroll
        for (uint32_t x=0; x<dim; x++) {
            result[x] += columnPtr[x]*vector[y];
        }
    }
}

template <typename real_t, uint32_t dim>
__device__ __forceinline__ void normalize(real_t* const __restrict__ vector)
{
    real_t rn;
    if(dim==3)
        rn = rnorm3d(vector[0],vector[1],vector[2]);
    else if(dim==2)
        rn = rhypot(vector[0],vector[1]);

    #pragma unroll
    for (int i=0;i<dim;i++) {
        vector[i] *= rn;
    }
}

template <typename real_t, uint32_t dim>
__device__ __forceinline__ void pointAt(const real_t* const __restrict__ ro,
                                        const real_t* const __restrict__ rd, 
                                        const real_t& delta, 
                                        real_t* const __restrict__ result)
{
    #pragma unroll
    for (int i=0;i<dim;i++) 
        result[i] = delta*rd[i]+ro[i];
}

template <typename real_t, uint32_t dim>
__device__ __forceinline__ void projectOntoBox(real_t* const __restrict__ point, 
                                               const uint32_t* const __restrict__ boxMax)
{
    #pragma unroll
    for (int i=0;i<dim;i++) {
        point[i] = point[i]<0 ? 0 : point[i];
        point[i] = point[i]>boxMax[i] ? boxMax[i] : point[i];
    }
        
}

template <typename real_t, uint32_t dim>
__device__ __forceinline__ bool closestVoxel(const real_t* const __restrict__ point,
                                             const uint32_t* const __restrict__ boxMax, 
                                             uint32_t* const __restrict__ voxelCoord, 
                                             const real_t* const __restrict__ rd, 
                                             const int* const __restrict__ stepDir)
{
    #pragma unroll
    for (int i=0;i<dim;i++) {
        // point has been projected onto box => point[i]>=0, can use uint32_t
        uint32_t fl = trunc(point[i]);
        voxelCoord[i] = fl == point[i] && rd[i]<0.0? fl-1 : fl;
        if (voxelCoord[i]>=boxMax[i])
            return false;
    }
    return true; 
}

template <typename real_t, uint32_t dim>
__device__ __forceinline__ void initStepDirection(const real_t* const __restrict__ rd,
                                                  int* const __restrict__ stepDir)
{
    #pragma unroll
    for (int i=0;i<dim;i++)
        stepDir[i] = ((rd[i]>0.0f) - (rd[i]<0.0f)); 
}

template <typename real_t, uint32_t dim>
__device__ __forceinline__ void initDelta(const real_t* const __restrict__ rd,
                                          const int* const __restrict__ stepDir,
                                          real_t* const __restrict__ delta) {
    #pragma unroll
    for (int i=0;i<dim;i++) {
        real_t d = stepDir[i]/rd[i];
        delta[i] = rd[i]>=-__FLT_EPSILON__ && rd[i]<=__FLT_EPSILON__ ? __FLT_MAX__ : d;
    }
}

template <typename real_t, uint32_t dim>
__device__ __forceinline__ void initMax(const real_t* const __restrict__ rd,
                                        const uint32_t* const __restrict__ currentVoxel,
                                        const real_t* const __restrict__ point,
                                        real_t* const __restrict__ tmax)
{
    real_t nextBoundary;
    #pragma unroll
    for (int i=0;i<dim;i++) {
        nextBoundary = rd[i]>0.0f ? currentVoxel[i] + 1 : currentVoxel[i];
        tmax[i] = rd[i]>=-__FLT_EPSILON__ && rd[i]<=__FLT_EPSILON__ ? __FLT_MAX__ : (nextBoundary - point[i])/rd[i];
    }
}

template <typename real_t, uint32_t dim>
__device__ __forceinline__ bool box_intersect(const real_t* const __restrict__ ro,
                                              const real_t* const __restrict__ rd,
                                              const uint32_t* const __restrict__ boxMax,
                                              real_t& tmin)
{
    real_t invDir = 1.0f / rd[0];

    real_t t1 = -ro[0] * invDir;
    real_t t2 = (boxMax[0] - ro[0]) * invDir;

    /**
     * fminf and fmaxf adhere to the IEEE standard, and return the non-NaN element if only a single
     * NaN is present
     */
    // tmin and tmax have to be picked for each specific direction without using fmin/fmax (supressing NaNs is bad in this case)
    tmin = invDir>=0 ? t1 : t2;
    real_t tmax = invDir>=0 ? t2 : t1;

    #pragma unroll
    for (int i = 1; i < dim; ++i)
    {
        invDir = 1.0f / rd[i];

        t1 = -ro[i] * invDir;
        t2 = (boxMax[i] - ro[i]) * invDir;

        tmin = fmax(tmin, invDir>=0 ? t1 : t2 );
        tmax = fmin(tmax, invDir>=0 ? t2 : t1 );
    }

    if (tmax == 0.0f && tmin == 0.0f)
        return false;
    if (tmax >= fmax(tmin, 0.0f)) // hit
        return true;
    return false;
}

template <typename real_t, uint32_t dim>
__device__ __forceinline__ uint32_t minIndex(const real_t* const __restrict__ tmax) {
    uint32_t index = 0;
    real_t min = tmax[0];

    #pragma unroll
    for (int i=1;i<dim;i++) {
        bool cond = tmax[i]<min;
        index = cond ? i : index;
        min = cond ? tmax[i] : min; 
    }

    return index;
}

template <typename real_t, uint32_t dim>
__device__ __forceinline__ bool isVoxelInVolume(const uint32_t* const __restrict__ currentVoxel,
                                                const uint32_t* const __restrict__ boxMax, 
                                                const uint32_t& index)
{
    return currentVoxel[index]<boxMax[index];
}

template <typename real_t, uint32_t dim>
__device__ __forceinline__ real_t updateTraverse(uint32_t* const __restrict__ currentVoxel, 
                                                 const int* const __restrict__ stepDir,
                                                 const real_t* const __restrict__ tdelta, 
                                                 real_t* const __restrict__ tmax, 
                                                 real_t& texit,
                                                 uint32_t& index)
{
    real_t tentry = texit;

    index = minIndex<real_t,dim>(tmax);
    texit = tmax[index];

    currentVoxel[index] += stepDir[index];
    tmax[index] += tdelta[index];

    return texit - tentry;
} 

/**
 * __restrict__ tells the compiler the arrays are not aliased, allowing for better optimization
 * must be true for all arrays
 */

template <typename real_t, bool adjoint, uint32_t dim>
__global__ void
traverse(int8_t* const __restrict__ volume,
        const uint64_t volumePitch,
        int8_t* const __restrict__ sinogram, 
        const uint64_t sinogramPitch,
        const int8_t* const __restrict__ rayOrigins,
        const uint32_t originPitch,
        const int8_t* const __restrict__ projInv,
        const uint32_t projPitch,
        const uint32_t* const __restrict__ boxMax)
{
    
    const int8_t* const projInvPtr = dim==3 ? projInv + (blockIdx.z*blockDim.x + threadIdx.x)*projPitch*3 : 
                                              projInv + (blockIdx.y*blockDim.x + threadIdx.x)*projPitch*2;

    const real_t* const rayOrigin = dim==3 ? (real_t*)(rayOrigins + (blockIdx.z*blockDim.x + threadIdx.x)*originPitch) :
                                             (real_t*)(rayOrigins + (blockIdx.y*blockDim.x + threadIdx.x)*originPitch);

    //homogenous pixel coordinates
    real_t pixelCoord[dim];
    pixelCoord[0] = blockIdx.x + 0.5f;
    pixelCoord[dim-1] = 1.0f;
    if(dim==3)
        pixelCoord[dim-2] = blockIdx.y + 0.5f;
    
    real_t* sinogramPtr = dim==3 ? (real_t*)(sinogram + ((blockIdx.z*blockDim.x+threadIdx.x)*gridDim.y + blockIdx.y)*sinogramPitch) + blockIdx.x 
                                    : (real_t*)(sinogram + (blockIdx.y*blockDim.x+threadIdx.x)*sinogramPitch) + blockIdx.x;
    
    if(!adjoint) *sinogramPtr = 0.0; 
   
    //compute ray direction
    real_t rd[dim]; 
    gesqmv<real_t, dim>(projInvPtr, pixelCoord, rd, projPitch);
    normalize<real_t, dim>(rd);

    //find volume intersections
    real_t tmin;
    if(!box_intersect<real_t,dim>(rayOrigin,rd,boxMax,tmin))
        return;

    real_t entryPoint[dim];
    pointAt<real_t,dim>(rayOrigin,rd,tmin,entryPoint);
    projectOntoBox<real_t,dim>(entryPoint,boxMax);
    
    int stepDir[dim];
    initStepDirection<real_t,dim>(rd,stepDir);

    uint32_t currentVoxel[dim];
    if(!closestVoxel<real_t,dim>(entryPoint, boxMax, currentVoxel, rd, stepDir))
        return;
    
    real_t tdelta[dim], tmax[dim];
    initDelta<real_t,dim>(rd,stepDir,tdelta);
    initMax<real_t,dim>(rd,currentVoxel,entryPoint,tmax);

    uint32_t index;
    real_t texit = 0.0;
    real_t pixelValue = 0.0; 
    
    real_t* volumeXPtr = dim==3 ? (real_t*)(volume + (boxMax[1]*currentVoxel[2] + currentVoxel[1])*volumePitch) + currentVoxel[0]
                                    : (real_t*)(volume + currentVoxel[1]*volumePitch) + currentVoxel[0];
    do {
        real_t d = updateTraverse<real_t,dim>(currentVoxel, stepDir, tdelta, tmax, texit, index);
        if (adjoint) 
            atomicAdd(volumeXPtr,*sinogramPtr*d);
        else
            pixelValue += d*(*volumeXPtr);
        
        volumeXPtr = dim==3 ? (real_t*)(volume + (boxMax[1]*currentVoxel[2] + currentVoxel[1])*volumePitch) + currentVoxel[0]
                                : (real_t*)(volume + currentVoxel[1]*volumePitch) + currentVoxel[0];
    } while(isVoxelInVolume<real_t,dim>(currentVoxel,boxMax,index));

    
    if (!adjoint) {
        *sinogramPtr = pixelValue;
    }
        
}


namespace elsa {
    template <typename data_t, uint32_t dim>
    void TraverseSiddonsCUDA<data_t,dim>::traverseForward(const dim3 blocks,const int threads,
              int8_t* const __restrict__ volume,
              const uint64_t volumePitch,
              int8_t* const __restrict__ sinogram,
              const uint64_t sinogramPitch,
              const int8_t* const __restrict__ rayOrigins,
              const uint32_t originPitch,
              const int8_t* const __restrict__ projInv,
              const uint32_t projPitch,
              const uint32_t* const __restrict__ boxMax,
              cudaStream_t stream) {
        traverse<data_t, false, dim><<<blocks,threads,0,stream>>>(volume,volumePitch,sinogram,sinogramPitch,rayOrigins,originPitch,projInv,projPitch,boxMax);
    }

    template <typename data_t, uint32_t dim>
    void TraverseSiddonsCUDA<data_t,dim>::traverseAdjoint(const dim3 blocks,const int threads,
              int8_t* const __restrict__ volume,
              const uint64_t volumePitch,
              int8_t* const __restrict__ sinogram,
              const uint64_t sinogramPitch,
              const int8_t* const __restrict__ rayOrigins,
              const uint32_t originPitch,
              const int8_t* const __restrict__ projInv,
              const uint32_t projPitch,
              const uint32_t* const __restrict__ boxMax,
              cudaStream_t stream) {
        traverse<data_t, true, dim><<<blocks,threads,0,stream>>>(volume,volumePitch,sinogram,sinogramPitch,rayOrigins,originPitch,projInv,projPitch,boxMax);
    }

    template struct TraverseSiddonsCUDA<float,2>;
    template struct TraverseSiddonsCUDA<float,3>;
}