#include "TraverseJosephsCUDA.cuh"

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
                                        const real_t delta, 
                                        real_t* const __restrict__ result)
{
    #pragma unroll
    for (int i=0;i<dim;i++) 
        result[i] = delta*rd[i]+ro[i];
}

template <typename real_t, uint32_t dim>
__device__ __forceinline__ void projectOntoBox(real_t* const __restrict__ point, 
                                               const typename elsa::TraverseJosephsCUDA<real_t,dim>::BoundingBox& boxMax) 
{
    #pragma unroll
    for (int i=0;i<dim;i++) {
        point[i] = point[i]<0.0f ? 0.0f : point[i];
        point[i] = point[i]>boxMax[i] ? boxMax[i] : point[i];
    }
        
}

template <typename real_t, uint32_t dim>
__device__ __forceinline__ bool closestVoxel(const real_t* const __restrict__ point,
                                             const typename elsa::TraverseJosephsCUDA<real_t,dim>::BoundingBox& boxMax, 
                                             uint32_t* const __restrict__ voxelCoord, 
                                             const real_t* const __restrict__ rd)
{
    #pragma unroll
    for (int i=0;i<dim;i++) {
        // point has been projected onto box => point[i]>=0, can use uint32_t
        uint32_t fl = trunc(point[i]);
        // for Joseph's also consider rays running along the "left" boundary
        voxelCoord[i] = fl == point[i] && rd[i]<=0.0f && point[i]!=0.0f ? fl-1 : fl;
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
                                              const typename elsa::TraverseJosephsCUDA<real_t,dim>::BoundingBox& boxMax,
                                              real_t& tmin,
                                              real_t& tmax)
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
    tmax = invDir>=0 ? t2 : t1;

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
__device__ __forceinline__ uint32_t minIndex(const real_t* const __restrict__ array) {
    uint32_t index = 0;
    real_t min = array[0];

    #pragma unroll
    for (int i=1;i<dim;i++) {
        bool cond = array[i]<min;
        index = cond ? i : index;
        min = cond ? array[i] : min; 
    }

    return index;
}

template <typename real_t, uint32_t dim>
__device__ __forceinline__ uint32_t maxAbsIndex(const real_t* const __restrict__ array) {
    uint32_t index = 0;
    real_t max = fabs(array[0]);

    #pragma unroll
    for (int i=1;i<dim;i++) {
        bool cond = fabs(array[i])>max;
        index = cond ? i : index;
        max = cond ? fabs(array[i]) : max; 
    }

    return index;
}

template <typename real_t, uint32_t dim>
__device__ __forceinline__ void updateTraverse(real_t* const __restrict__ currentPosition, 
                                                 const real_t* const __restrict__ rd, 
                                                 const real_t dist)
{
    #pragma unroll
    for (uint32_t i=0;i<dim;i++)
        currentPosition[i]+=rd[i]*dist;
} 

template <typename real_t, uint dim>
__device__ __forceinline__ real_t tex(cudaTextureObject_t texObj, const real_t* const p) {
    if(dim==3)
        return tex3D<real_t>(texObj, p[0],p[1],p[2]);
    else
        return tex2D<real_t>(texObj, p[0],p[1]);
}


template <typename real_t, uint dim>
__global__ void __launch_bounds__(elsa::TraverseJosephsCUDA<real_t,dim>::MAX_THREADS_PER_BLOCK)
    traverseForwardKernel(cudaTextureObject_t volume,
                    int8_t* const __restrict__ sinogram,
                    const uint64_t sinogramPitch,
                    const int8_t* const __restrict__ rayOrigins,
                    const uint32_t originPitch,
                    const int8_t* const __restrict__ projInv,
                    const uint32_t projPitch,
                    const typename elsa::TraverseJosephsCUDA<real_t,dim>::BoundingBox boxMax) {

    const int8_t* const projInvPtr = dim==3 ? projInv + (blockIdx.z*blockDim.x + threadIdx.x)*projPitch*3 : 
                                              projInv + (blockIdx.y*blockDim.x + threadIdx.x)*projPitch*2;

    const real_t* const rayOrigin = dim==3 ? (real_t*)(rayOrigins + (blockIdx.z*blockDim.x + threadIdx.x)*originPitch) :
                                             (real_t*)(rayOrigins + (blockIdx.y*blockDim.x + threadIdx.x)*originPitch);

    real_t* sinogramPtr = dim==3 ? ((real_t*)(sinogram + ((blockIdx.z*blockDim.x+threadIdx.x)*gridDim.y + blockIdx.y)*sinogramPitch) + blockIdx.x) :
                                        ((real_t*)(sinogram + (blockIdx.y*blockDim.x + threadIdx.x)*sinogramPitch)+blockIdx.x);
    
    *sinogramPtr = 0;

    //homogenous pixel coordinates
    real_t pixelCoord[dim];
    pixelCoord[0] = blockIdx.x + 0.5f;
    pixelCoord[dim-1] = 1.0f;
    if(dim==3)
      pixelCoord[dim-2]=blockIdx.y + 0.5f;
        
    
    //compute ray direction
    real_t rd[dim];
    gesqmv<real_t, dim>(projInvPtr, pixelCoord, rd, projPitch);
    normalize<real_t, dim>(rd);

    //find volume intersections
    real_t tmin, tmax;
    if(!box_intersect<real_t,dim>(rayOrigin,rd,boxMax,tmin,tmax))
        return;
    
    real_t currentPosition[dim];
    pointAt<real_t,dim>(rayOrigin,rd,tmin,currentPosition);

    //determine primary direction
    const uint32_t idx = maxAbsIndex<real_t,dim>(rd);
    const real_t tdelta = 1.0f/fabs(rd[idx]);

    // truncate as currentPosition is non-negative
    const real_t fl = trunc(currentPosition[idx]);
    // for Joseph's also consider rays running along the "left" boundary
    const real_t firstBoundary = fl == currentPosition[idx] && rd[idx]<0.0f ? fl-1.0f : fl;

    //find distance to next plane orthogonal to primary diretion
    const real_t nextBoundary = rd[idx]>0.0f ? firstBoundary + 1.0f : firstBoundary;
    real_t minDelta = (nextBoundary - currentPosition[idx])/rd[idx];

    real_t intersectionLength = tmax - tmin;
    //first plane intersection may lie outside the bounding box
    if(intersectionLength<minDelta) {
        //use midpoint of entire ray intersection as a constant integration value
        updateTraverse<real_t,dim>(currentPosition,rd,intersectionLength*0.5f);

        *sinogramPtr = intersectionLength*tex<real_t,dim>(volume,currentPosition);
        return;
    }

    /**
     * otherwise first plane intersection inside bounding box
     * add first line segment and move to first interior point
     */
    updateTraverse<real_t,dim>(currentPosition,rd,minDelta*0.5f);
    real_t pixelValue = minDelta*tex<real_t,dim>(volume,currentPosition);

    //from here on use tmin as an indication of the current position along the ray
    tmin+=minDelta;

    //if next point isn't last
    if (tmax-tmin>tdelta){
        updateTraverse<real_t,dim>(currentPosition,rd,(minDelta+tdelta)*0.5f);
        tmin+=tdelta;
        minDelta = tdelta;
        pixelValue += minDelta*tex<real_t,dim>(volume,currentPosition);

        //while interior intersection points remain
        while (tmin+minDelta<tmax) {                
            updateTraverse<real_t,dim>(currentPosition, rd, minDelta);
            tmin+=minDelta;
            pixelValue += minDelta*tex<real_t,dim>(volume,currentPosition);
        }
    }

    updateTraverse<real_t,dim>(currentPosition,rd,(tmax+minDelta-tmin)*0.5f);
    pixelValue += (tmax-tmin)*tex<real_t,dim>(volume,currentPosition);

    *sinogramPtr = pixelValue;
}

//TODO: check if sorting can be used to make this even faster
template <typename real_t, uint32_t dim>
__global__  void __launch_bounds__(elsa::TraverseJosephsCUDA<real_t,dim>::MAX_THREADS_PER_BLOCK)
    traverseAdjointFastKernel(int8_t* const __restrict__ volume,
                        const uint64_t volumePitch,
                        cudaTextureObject_t sinogram,
                        const int8_t* const __restrict__ rayOrigins,
                        const uint32_t originPitch,
                        const int8_t* const __restrict__ proj,
                        const uint32_t projPitch,
                        const uint32_t numAngles,
                        const uint32_t offset) {
    int x = blockIdx.x;
    int y = dim==3 ? blockIdx.y : blockIdx.y*blockDim.x + threadIdx.x;
    int z = dim==3 ? blockIdx.z*blockDim.x + threadIdx.x : 0;

    real_t& voxelRef = *(real_t*)(volume + x*sizeof(real_t) + y*volumePitch + z*volumePitch*gridDim.y);
        
    real_t voxelCenter[dim];
    voxelCenter[0] = x+0.5f;
    if (dim==3) {
        voxelCenter[dim-2] = y+0.5f;
        voxelCenter[dim-1] = z+offset+0.5f;
    }
    else {
        voxelCenter[1] = y+offset+0.5f;
    }

    real_t val = 0.0f;
    for (uint i=0;i<numAngles;i++) {
        const int8_t* const projPtr = proj + i*projPitch*dim;
        const real_t* const rayOrigin = (real_t*)(rayOrigins + i*originPitch); 
        
        //compute ray direction
        real_t rd[dim];
        #pragma unroll
        for (uint j=0;j<dim;j++)
            rd[j]=voxelCenter[j]-rayOrigin[j];

        real_t pixelCoord[dim];

        gesqmv<real_t, dim>(projPtr, rd, pixelCoord, projPitch);

        //convert to homogenous coordinates
        pixelCoord[0]/=pixelCoord[dim-1];
        
        if (dim==3) {
            pixelCoord[1]/=pixelCoord[dim-1];
            val += tex2DLayered<real_t>(sinogram,pixelCoord[0],pixelCoord[1],i);
        }
        else {
            val += tex1DLayered<real_t>(sinogram,pixelCoord[0],i);
        }

    }

    voxelRef = val;
}

template <typename real_t,uint dim>
__device__ __forceinline__ void backproject2(int8_t* const __restrict__ volume,
                                             const uint64_t* const __restrict__ p, 
                                             uint32_t* const __restrict__ voxelCoord,
                                             const real_t* const __restrict__ voxelCoordf,
                                             const typename elsa::TraverseJosephsCUDA<real_t,dim>::BoundingBox& boxMax,
                                             const real_t* const __restrict__ frac,
                                             const real_t weightedVal) {
    
    real_t* volumeXPtr = (real_t*)(volume + p[0]*voxelCoord[0] + p[1]*voxelCoord[1]);
    real_t val = (1.0f-frac[1])*weightedVal;
    atomicAdd(volumeXPtr,val);

    //volume[i,j+1]
    voxelCoord[1] = voxelCoord[1]<boxMax[1]-1 ? voxelCoordf[1] + 1 : boxMax[1] - 1;
    volumeXPtr = (real_t*)(volume + p[0]*voxelCoord[0] + p[1]*voxelCoord[1]);
    val = frac[1]*weightedVal;
    atomicAdd(volumeXPtr, val);
}

template <typename real_t,uint dim>
__device__ __forceinline__ void backproject4(int8_t* const __restrict__ volume,
                                             const uint64_t* const __restrict__ p, 
                                             uint32_t* const __restrict__ voxelCoord,
                                             const real_t* const __restrict__ voxelCoordf,
                                             const typename elsa::TraverseJosephsCUDA<real_t,dim>::BoundingBox& boxMax,
                                             const real_t* const __restrict__ frac,
                                             const real_t weightedVal) {    
    real_t* volumeXPtr = (real_t*)(volume + p[0]*voxelCoord[0] + p[1]*voxelCoord[1] + p[2]*voxelCoord[2]);
    real_t val = (1.0f-frac[1])*(1.0f-frac[2])*weightedVal;
    atomicAdd(volumeXPtr,val);
    //frac[0] is 0

    //volume[i,j+1,k]
    voxelCoord[1] = voxelCoord[1]<boxMax[1]-1.0f ? voxelCoordf[1] + 1.0f : boxMax[1] - 1.0f;
    volumeXPtr = (real_t*)(volume + p[0]*voxelCoord[0] + p[1]*voxelCoord[1] + p[2]*voxelCoord[2]);
    val = frac[1]*(1.0f-frac[2])*weightedVal;
    atomicAdd(volumeXPtr, val);

    //volume[i,j+1,k+1]
    voxelCoord[2] = voxelCoord[2]<boxMax[2]-1.0f ? voxelCoordf[2] + 1.0f : boxMax[2] - 1.0f;
    volumeXPtr = (real_t*)(volume + p[0]*voxelCoord[0] + p[1]*voxelCoord[1] + p[2]*voxelCoord[2]);
    val = frac[1]*frac[2]*weightedVal;
    atomicAdd(volumeXPtr, val);

    //volume[i,j,k+1]
    voxelCoord[1] = voxelCoordf[1]<0.0f ? 0 : voxelCoordf[1];
    volumeXPtr = (real_t*)(volume + p[0]*voxelCoord[0] + p[1]*voxelCoord[1] + p[2]*voxelCoord[2]);
    val = (1.0f-frac[1])*frac[2]*weightedVal;
    atomicAdd(volumeXPtr, val);
}

template <typename real_t, uint dim>
__device__ __forceinline__ void backproject(int8_t* const __restrict__ volume,
                                            const uint64_t* const __restrict__ p, 
                                            uint32_t* const __restrict__ voxelCoord,
                                            const real_t* const __restrict__ voxelCoordf,
                                            const typename elsa::TraverseJosephsCUDA<real_t,dim>::BoundingBox& boxMax,
                                            const real_t* const __restrict__ frac,
                                            const real_t weightedVal) {
    if (dim == 3)
        backproject4<real_t,dim>(volume,p,voxelCoord,voxelCoordf,boxMax,frac,weightedVal);
    else 
        backproject2<real_t,dim>(volume,p,voxelCoord,voxelCoordf,boxMax,frac,weightedVal);
}

template <typename T>
__device__ __forceinline__ void swap(T& a, T& b) {
    T c = a;
    a = b;
    b = c;
}

template <typename real_t, uint dim>
__global__  void __launch_bounds__(elsa::TraverseJosephsCUDA<real_t,dim>::MAX_THREADS_PER_BLOCK) 
    traverseAdjointKernel(int8_t* const __restrict__ volume,
                    const uint64_t volumePitch,
                    const int8_t* const __restrict__ sinogram,
                    const uint64_t sinogramPitch,
                    const int8_t* const __restrict__ rayOrigins,
                    const uint32_t originPitch,
                    const int8_t* const __restrict__ projInv,
                    const uint32_t projPitch,
                    typename elsa::TraverseJosephsCUDA<real_t,dim>::BoundingBox boxMax) {

    const int8_t* const projInvPtr = dim==3 ? projInv + (blockIdx.z*blockDim.x + threadIdx.x)*projPitch*3 : 
                                              projInv + (blockIdx.y*blockDim.x + threadIdx.x)*projPitch*2;

    const real_t* const rayOrigin = dim==3 ? (real_t*)(rayOrigins + (blockIdx.z*blockDim.x + threadIdx.x)*originPitch) :
                                             (real_t*)(rayOrigins + (blockIdx.y*blockDim.x + threadIdx.x)*originPitch);

    const real_t sinogramVal = dim==3 ? *((real_t*)(sinogram + ((blockIdx.z*blockDim.x+threadIdx.x)*gridDim.y + blockIdx.y)*sinogramPitch) + blockIdx.x):
                                        *((real_t*)(sinogram + (blockIdx.y*blockDim.x + threadIdx.x)*sinogramPitch)+blockIdx.x);

    //homogenous pixel coordinates
    real_t pixelCoord[dim]; 
    pixelCoord[0] = blockIdx.x + 0.5f;
    pixelCoord[dim-1] = 1.0f;
    if(dim==3)
       pixelCoord[1] = blockIdx.y + 0.5f;
        
    //compute ray direction
    real_t rd[dim];
    gesqmv<real_t, dim>(projInvPtr, pixelCoord, rd, projPitch);
    normalize<real_t, dim>(rd);
    
    //find volume intersections
    real_t tmin, tmax;
    if(!box_intersect<real_t,dim>(rayOrigin,rd,boxMax,tmin,tmax))
        return;
    
    int stepDir[dim];
    real_t tdelta[dim];
    initStepDirection<real_t,dim>(rd,stepDir);
    initDelta<real_t,dim>(rd,stepDir,tdelta);
    
    real_t currentPosition[dim];
    pointAt<real_t,dim>(rayOrigin,rd,tmin,currentPosition);
    projectOntoBox<real_t,dim>(currentPosition,boxMax);

    uint32_t voxelCoord[dim];
    if(!closestVoxel<real_t,dim>(currentPosition,boxMax,voxelCoord,rd)) return;

    //determine primary direction
    uint32_t idx = minIndex<real_t,dim>(tdelta);
    const int s = stepDir[idx];

    uint64_t permutation[dim];
    permutation[0] = sizeof(real_t);
    permutation[1] = volumePitch;
    if (dim==3)
        permutation[dim-1] = volumePitch*boxMax[1];
    
    //find distance to next plane orthogonal to primary diretion
    real_t nextBoundary = rd[idx]>0.0f ? voxelCoord[idx] + 1 : voxelCoord[idx];
    real_t minDelta = (nextBoundary - currentPosition[idx])/rd[idx];

    uint32_t entryDir = 0;
    for (uint i=1;i<dim;i++)
        //current position is already projected onto the box, so a direct comparison works
        if(currentPosition[i]==0 || currentPosition[i]==boxMax[i])
            entryDir = i;

    real_t intersectionLength = tmax - tmin;

    real_t voxelCoordf[dim], frac[dim];

    //subtract 0.5 from current position to get voxel coordinates
    for (uint i=0;i<dim;i++) {
        currentPosition[i]-=0.5f;
    }

    //permute indices, so that entry direction is at first index
    swap<real_t>(rd[0],rd[entryDir]);
    swap<real_t>(currentPosition[0],currentPosition[entryDir]);
    swap<uint32_t>(voxelCoord[0],voxelCoord[entryDir]);
    swap<real_t>(tdelta[0],tdelta[entryDir]);
    swap<real_t>(boxMax[0],boxMax[entryDir]);
    swap<uint64_t>(permutation[0],permutation[entryDir]);

    //first plane intersection may lie outside the bounding box
    if(intersectionLength<minDelta) {
        //use midpoint of entire ray intersection with bounding box as a constant integration value
        updateTraverse<real_t,dim>(currentPosition,rd,intersectionLength*0.5f);
        for (uint i=0;i<dim;i++) {
            voxelCoordf[i] = floorf(currentPosition[i]);
            frac[i] = currentPosition[i]-voxelCoordf[i];
            voxelCoord[i] =  fmax(voxelCoordf[i],static_cast<real_t>(0));
        }
        backproject<real_t,dim>(volume,permutation,voxelCoord,voxelCoordf,boxMax,frac,intersectionLength*sinogramVal);
        return;
    }

    
    /**
     * otherwise first plane intersection inside bounding box
     * add first line segment and move to first interior point
     */
    updateTraverse<real_t,dim>(currentPosition,rd,minDelta*0.5f);
    for (uint i=0;i<dim;i++) {
        voxelCoordf[i] = floorf(currentPosition[i]);
        frac[i] = currentPosition[i]-voxelCoordf[i];
        voxelCoord[i] =  fmax(voxelCoordf[i],static_cast<real_t>(0));
    }
    backproject<real_t,dim>(volume,permutation,voxelCoord,voxelCoordf,boxMax,frac,minDelta*sinogramVal);
    //from here on use tmin as an indication of the current position along the ray
    tmin+=minDelta;

    idx = minIndex<real_t,dim>(tdelta);
    //permute indices, so that primary direction is at first index
    swap<real_t>(rd[0],rd[idx]);
    swap<real_t>(currentPosition[0],currentPosition[idx]);
    swap<uint32_t>(voxelCoord[0],voxelCoord[idx]);
    swap<real_t>(tdelta[0],tdelta[idx]);
    swap<real_t>(boxMax[0],boxMax[idx]);
    swap<uint64_t>(permutation[0],permutation[idx]);

    //if next point isn't last
    if (tmax-tmin>tdelta[0]){
        updateTraverse<real_t,dim>(currentPosition,rd,(minDelta+tdelta[0])*0.5f);
        minDelta = tdelta[0];
        tmin+=minDelta;
        
        //set up values at idx manually, might lead to errors else
        currentPosition[0] = round(currentPosition[0]);
        frac[0] = 0.0f;
        voxelCoord[0] = (uint32_t)currentPosition[0];

        for (uint i=1;i<dim;i++) {
            voxelCoordf[i] = floorf(currentPosition[i]);
            frac[i] = currentPosition[i]-voxelCoordf[i];
            voxelCoord[i] = fmax(static_cast<real_t>(0),voxelCoordf[i]);
        }
        backproject<real_t,dim>(volume,permutation,voxelCoord,voxelCoordf,boxMax,frac,minDelta*sinogramVal);

        //while interior intersection points remain
        while (tmin+minDelta<tmax) {                
            updateTraverse<real_t,dim>(currentPosition, rd, minDelta);
            tmin+=minDelta;

            voxelCoord[0]+=s;
            for (uint i=1;i<dim;i++) {
                voxelCoordf[i] = floorf(currentPosition[i]);
                frac[i] = currentPosition[i]-voxelCoordf[i];
                voxelCoord[i] = fmax(voxelCoordf[i],static_cast<real_t>(0));
            }
            backproject<real_t,dim>(volume,permutation,voxelCoord,voxelCoordf,boxMax,frac,minDelta*sinogramVal);

        }
    }

    updateTraverse<real_t,dim>(currentPosition,rd, (tmax+minDelta-tmin)*0.5f );
    for (uint32_t i=1;i<dim;i++) {
        // for large volumes numerical errors sometimes cause currentPosition of the last voxel
        // to lie outside boxMax although ideally it should not even exceed boxMax-0.5; currently 
        // solved by readjusting the coordinates if needed
        // TODO: try updating the traversal using pointAt() instead
        voxelCoordf[i] = floorf(currentPosition[i]);
        frac[i] = currentPosition[i]-voxelCoordf[i];
        voxelCoord[i] =  fmax(voxelCoordf[i],static_cast<real_t>(0));

        if(voxelCoord[i]>=boxMax[i]) {
            voxelCoord[i] = boxMax[i]-1.0f;
            frac[i] = 0.5f;
        }
    }

    real_t mainDirPos = currentPosition[0];
    for (uint32_t i=0;i<dim;i++) {
        // move to exit point
        currentPosition[i] = currentPosition[i] + 0.5f +rd[i]*(tmax-tmin)*0.5f;

        // distance to border
        currentPosition[i] = fabs(fmin(currentPosition[i], boxMax[i]-currentPosition[i]));
    }

    // find direction closest to border
    uint32_t exitDir = minIndex<real_t,dim>(currentPosition);
    
    if(exitDir == 0) {
        //again handle this case manually
        voxelCoordf[0] = (real_t)voxelCoord[0] + s;
        voxelCoord[0] = fmin(fmax(voxelCoordf[0],static_cast<real_t>(0)),boxMax[0]-static_cast<real_t>(1));
        frac[0] = mainDirPos-voxelCoordf[0];
    }
    else {
        voxelCoordf[0] = floorf(mainDirPos);
        frac[0] = mainDirPos-voxelCoordf[0];
        voxelCoord[0] =  fmax(static_cast<real_t>(0),voxelCoordf[0]);
    }
    
    //permute indices, so that exit direction is at first index
    swap<uint32_t>(voxelCoord[0],voxelCoord[exitDir]);
    swap<real_t>(voxelCoordf[0],voxelCoordf[exitDir]);
    swap<real_t>(frac[0],frac[exitDir]);
    swap<real_t>(boxMax[0],boxMax[exitDir]);
    swap<uint64_t>(permutation[0],permutation[exitDir]);
    backproject<real_t,dim>(volume,permutation,voxelCoord,voxelCoordf,boxMax,frac,(tmax-tmin)*sinogramVal); 
}

namespace elsa {

template <typename data_t,uint32_t dim> 
void TraverseJosephsCUDA<data_t,dim>::traverseForward(const dim3 blocks, const int threads,
                    cudaTextureObject_t volume,
                    int8_t* const __restrict__ sinogram,
                    const uint64_t sinogramPitch,
                    const int8_t* const __restrict__ rayOrigins,
                    const uint32_t originPitch,
                    const int8_t* const __restrict__ projInv,
                    const uint32_t projPitch,
                    const BoundingBox boxMax,
                    const cudaStream_t stream) {
    traverseForwardKernel<data_t,dim><<<blocks,threads,0,stream>>>(volume,sinogram,sinogramPitch,rayOrigins,originPitch,projInv,projPitch,boxMax);
}

template <typename data_t, uint32_t dim>
void TraverseJosephsCUDA<data_t,dim>::traverseAdjoint(const dim3 blocks, const int threads,
                    int8_t* const __restrict__ volume,
                    const uint64_t volumePitch,
                    const int8_t* const __restrict__ sinogram,
                    const uint64_t sinogramPitch,
                    const int8_t* const __restrict__ rayOrigins,
                    const uint32_t originPitch,
                    const int8_t* const __restrict__ projInv,
                    const uint32_t projPitch,
                    BoundingBox boxMax,
                    const cudaStream_t stream) {
    traverseAdjointKernel<data_t,dim><<<blocks,threads,0,stream>>>(volume, volumePitch, sinogram, sinogramPitch, rayOrigins, originPitch, projInv, projPitch, boxMax);
}

template <typename data_t, uint32_t dim>
void TraverseJosephsCUDA<data_t,dim>::traverseAdjointFast(const dim3 blocks, const int threads,
                        int8_t* const __restrict__ volume,
                        const uint64_t volumePitch,
                        cudaTextureObject_t sinogram,
                        const int8_t* const __restrict__ rayOrigins,
                        const uint32_t originPitch,
                        const int8_t* const __restrict__ proj,
                        const uint32_t projPitch,
                        const uint32_t numAngles,
                        const uint32_t zOffset,
                        const cudaStream_t stream) {
    traverseAdjointFastKernel<data_t,dim><<<blocks,threads,0,stream>>>(volume,volumePitch,sinogram,rayOrigins,originPitch,proj,projPitch,numAngles,zOffset);
}

//template instantiations
template struct TraverseJosephsCUDA<float,2>;
template struct TraverseJosephsCUDA<float,3>;
}