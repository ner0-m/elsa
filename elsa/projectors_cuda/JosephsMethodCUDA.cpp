#include "JosephsMethodCUDA.h"
#include "LogGuard.h"
#include "Timer.h"

namespace elsa
{
    template <typename data_t>
    JosephsMethodCUDA<data_t>::JosephsMethodCUDA(const DataDescriptor &domainDescriptor, const DataDescriptor &rangeDescriptor,
                               const std::vector<Geometry> &geometryList, bool fast)
            : LinearOperator<data_t>(domainDescriptor, rangeDescriptor), _geometryList{geometryList},
              _boundingBox{_domainDescriptor->getNumberOfCoefficientsPerDimension()}, _fast{fast}
    {
        auto dim = _domainDescriptor->getNumberOfDimensions();
        if (dim != _rangeDescriptor->getNumberOfDimensions()) {
            throw std::logic_error(std::string("JosephsMethodCUDA: domain and range dimension need to match"));
        }

        if (dim!=2 && dim != 3) {
            throw std::logic_error("JosephsMethodCUDA: only supporting 2d/3d operations");
        }

        if (_geometryList.empty()) {
            throw std::logic_error("JosephsMethodCUDA: geometry list was empty");
        }

        //allocate device memory and copy ray origins and the inverse of projection matrices to device
        cudaExtent extent = make_cudaExtent(dim*sizeof(real_t), dim, _geometryList.size());

        if(cudaMallocPitch(&_rayOrigins.ptr, &_rayOrigins.pitch,dim*sizeof(real_t),_geometryList.size())!=cudaSuccess)
            throw std::bad_alloc();
        _rayOrigins.xsize = dim;
        _rayOrigins.ysize = _geometryList.size();

        
        if (cudaMalloc3D(&_projInvMatrices, extent)!=cudaSuccess)
            throw std::bad_alloc();
        
        if (fast)
            if (cudaMalloc3D(&_projMatrices, extent)!=cudaSuccess)
                throw std::bad_alloc();

        index_t projPitch = _projInvMatrices.pitch;
        int8_t* rayBasePtr = (int8_t*)_rayOrigins.ptr;
        index_t rayPitch = _rayOrigins.pitch;
        for (index_t i=0;i<_geometryList.size();i++) {
            RealMatrix_t P = _geometryList[i].getInverseProjectionMatrix().block(0,0,dim,dim);
            int8_t* slice = (int8_t*)_projInvMatrices.ptr + i*projPitch*dim;

            //transfer inverse of projection matrix
            if (cudaMemcpy2D(slice, projPitch, P.data(), dim*sizeof(real_t), dim*sizeof(real_t), dim, cudaMemcpyHostToDevice) != cudaSuccess)
                throw std::logic_error("JosephsMethodCUDA: Could not transfer inverse projection matrices to GPU.");
            
            //transfer projection matrix if _fast flag is set
            if (_fast) {
                P = _geometryList[i].getProjectionMatrix().block(0,0,dim,dim);
                slice = (int8_t*)_projMatrices.ptr + i*projPitch*dim;
                if (cudaMemcpy2D(slice, projPitch, P.data(), dim*sizeof(real_t), dim*sizeof(real_t), dim, cudaMemcpyHostToDevice) != cudaSuccess)
                    throw std::logic_error("JosephsMethodCUDA: Could not transfer inverse projection matrices to GPU.");
            }
            
            int8_t* rayPtr = rayBasePtr + i*rayPitch;
            // get ray origin using direct inverse
            RealVector_t ro = -_geometryList[i].getInverseProjectionMatrix().block(0,0,dim,dim)*_geometryList[i].getProjectionMatrix().block(0,dim,dim,1);
            //transfer ray origin
            if (cudaMemcpyAsync(rayPtr,ro.data(),dim*sizeof(real_t),cudaMemcpyHostToDevice)!=cudaSuccess)
                throw std::logic_error("JosephsMethodCUDA: Could not transfer ray origins to GPU.");
            
        }
 
    }


    template <typename data_t>
    JosephsMethodCUDA<data_t>::~JosephsMethodCUDA() 
    {
        //Free CUDA resources
        if (cudaFree(_rayOrigins.ptr)!=cudaSuccess ||
            cudaFree(_projInvMatrices.ptr)!=cudaSuccess)
                Logger::get("JosephsMethodCUDA")->error("Couldn't free GPU memory; This may cause problems later.");
        
        if (_fast)
            if (cudaFree(_projMatrices.ptr)!=cudaSuccess)
                Logger::get("JosephsMethodCUDA")->error("Couldn't free GPU memory; This may cause problems later.");
    }

    template <typename data_t>
    JosephsMethodCUDA<data_t>* JosephsMethodCUDA<data_t>::cloneImpl() const
    {
        //TODO: Currently only need Geometry vector stored internally for cloning, try removing it 
        return new JosephsMethodCUDA(*_domainDescriptor, *_rangeDescriptor,_geometryList,_fast);
    }

    template <typename data_t>
    bool JosephsMethodCUDA<data_t>::isEqual(const LinearOperator<data_t>& other) const 
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        auto otherJM = dynamic_cast<const JosephsMethodCUDA*>(&other);
        if (!otherJM)
            return false;

        if (_geometryList != otherJM->_geometryList || _fast != otherJM->_fast)
            return false;
        
        if (_fast != otherJM->_fast)
            return false;

        return true;
    }

    template <typename data_t>
    void JosephsMethodCUDA<data_t>::_apply(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const
    {
        Timer<> timeGuard("JosephsMethodCUDA", "apply");

        //transfer volume as texture
        auto [volumeTex, volume] = copyTextureToGPU(x);

        // allocate memory for projections
        cudaPitchedPtr dsinoPtr;
        index_t dim = _domainDescriptor->getNumberOfDimensions();
        IndexVector_t rangeDims = _rangeDescriptor->getNumberOfCoefficientsPerDimension();

        int numImgBlocks = rangeDims(dim-1)/_threadsPerBlock;
        int remaining = rangeDims(dim-1)%_threadsPerBlock;

        if(dim == 3) {
            cudaExtent sinoExt = make_cudaExtent(rangeDims[0]*sizeof(data_t),rangeDims[1],rangeDims[2]);
            if(cudaMalloc3D(&dsinoPtr,sinoExt)!=cudaSuccess)
                throw std::bad_alloc();
            
            IndexVector_t bmax = _boundingBox._max.template cast<index_t>();
            typename TraverseJosephsCUDA<data_t,3>::BoundingBox boxMax;
            boxMax.max[0] = bmax[0];
            boxMax.max[1] = bmax[1];
            boxMax.max[2] = bmax[2];

            //synchronize because we are using multiple streams
            cudaDeviceSynchronize();

            //perform projection
            if (numImgBlocks>0) {
                const dim3 grid(rangeDims(0),rangeDims(1),numImgBlocks);
                const int threads = _threadsPerBlock;
                TraverseJosephsCUDA<data_t,3>::traverseForward(grid, threads,volumeTex,(int8_t*)dsinoPtr.ptr, dsinoPtr.pitch,(int8_t*)_rayOrigins.ptr,_rayOrigins.pitch,(int8_t*)_projInvMatrices.ptr,_projInvMatrices.pitch,boxMax);        
            }

            if (remaining>0) {
                cudaStream_t remStream;
                if(cudaStreamCreate(&remStream)!=cudaSuccess)
                    throw std::logic_error("JosephsMethodCUDA::apply: Couldn't create stream for remaining images");
                const dim3 grid(rangeDims(0),rangeDims(1),1);
                const int threads = remaining;
                int8_t* imgPtr = (int8_t*)dsinoPtr.ptr + numImgBlocks*_threadsPerBlock*dsinoPtr.pitch*rangeDims(1);
                int8_t* rayPtr = (int8_t*)_rayOrigins.ptr + numImgBlocks*_threadsPerBlock*_rayOrigins.pitch;
                int8_t* matrixPtr = (int8_t*)_projInvMatrices.ptr + numImgBlocks*_threadsPerBlock*_projInvMatrices.pitch*dim;
                TraverseJosephsCUDA<data_t,3>::traverseForward(grid, threads,volumeTex,imgPtr, dsinoPtr.pitch,rayPtr,_rayOrigins.pitch,matrixPtr,_projInvMatrices.pitch,boxMax,remStream);

                if (cudaStreamDestroy(remStream)!=cudaSuccess) 
                    Logger::get("JosephsMethodCUDA")->error("Couldn't destroy CUDA stream; This may cause problems later.");
            }

            cudaDeviceSynchronize();
            //retrieve results from GPU
            copy3DDataContainer<cudaMemcpyDeviceToHost,false>((void*)&Ax[0],dsinoPtr,sinoExt);
        }
        else {
            if (cudaMallocPitch(&dsinoPtr.ptr,&dsinoPtr.pitch,rangeDims[0]*sizeof(data_t),rangeDims[1])!=cudaSuccess)
                throw std::bad_alloc();
            
            IndexVector_t bmax = _boundingBox._max.template cast<index_t>();
            typename TraverseJosephsCUDA<data_t,2>::BoundingBox boxMax;
            boxMax.max[0] = bmax[0];
            boxMax.max[1] = bmax[1];

            //synchronize because we are using multiple streams
            cudaDeviceSynchronize();

            //perform projection
            if (numImgBlocks>0) {
                const dim3 grid(rangeDims(0),numImgBlocks);
                const int threads = _threadsPerBlock;
                TraverseJosephsCUDA<data_t,2>::traverseForward(grid, threads,volumeTex,(int8_t*)dsinoPtr.ptr, dsinoPtr.pitch,(int8_t*)_rayOrigins.ptr,_rayOrigins.pitch,(int8_t*)_projInvMatrices.ptr,_projInvMatrices.pitch,boxMax);        
            }

            if (remaining>0) {
                cudaStream_t remStream;
                if(cudaStreamCreate(&remStream)!=cudaSuccess)
                    throw std::logic_error("JosephsMethodCUDA::apply: Couldn't create stream for remaining images");
                const dim3 grid(rangeDims(0),1);
                const int threads = remaining;
                int8_t* imgPtr = (int8_t*)dsinoPtr.ptr + numImgBlocks*_threadsPerBlock*dsinoPtr.pitch;
                int8_t* rayPtr = (int8_t*)_rayOrigins.ptr + numImgBlocks*_threadsPerBlock*_rayOrigins.pitch;
                int8_t* matrixPtr = (int8_t*)_projInvMatrices.ptr + numImgBlocks*_threadsPerBlock*_projInvMatrices.pitch*dim;
                TraverseJosephsCUDA<data_t,2>::traverseForward(grid, threads,volumeTex,imgPtr, dsinoPtr.pitch,rayPtr,_rayOrigins.pitch,matrixPtr,_projInvMatrices.pitch,boxMax,remStream);

                if (cudaStreamDestroy(remStream)!=cudaSuccess) 
                    Logger::get("JosephsMethodCUDA")->error("Couldn't destroy CUDA stream; This may cause problems later.");
            }

            cudaDeviceSynchronize();
            //retrieve results from GPU
            if(cudaMemcpy2D((void*)&Ax[0],rangeDims[0]*sizeof(data_t),dsinoPtr.ptr,dsinoPtr.pitch,rangeDims[0]*sizeof(data_t),rangeDims[1],cudaMemcpyDeviceToHost)!=cudaSuccess)
                throw std::logic_error("JosephsMethodCUDA::apply: Couldn't retrieve results from GPU");
        }

        if (cudaDestroyTextureObject(volumeTex)!=cudaSuccess)
            Logger::get("JosephsMethodCUDA")->error("Couldn't destroy texture object; This may cause problems later.");

        if(cudaFreeArray(volume)!=cudaSuccess)
            Logger::get("JosephsMethodCUDA")->error("Couldn't free GPU memory; This may cause problems later.");

        if (cudaFree(dsinoPtr.ptr) != cudaSuccess)
            Logger::get("JosephsMethodCUDA")->error("Couldn't free GPU memory; This may cause problems later.");
    }

    template <typename data_t>
    void JosephsMethodCUDA<data_t>::_applyAdjoint(const DataContainer<data_t>& y, DataContainer<data_t>& Aty) const
    {
        Timer<> timeguard("JosephsMethodCUDA", "applyAdjoint");

        //allocate memory for volume
        auto coeffsPerDim =  _domainDescriptor->getNumberOfCoefficientsPerDimension();
        index_t dim = _domainDescriptor->getNumberOfDimensions();

        int numImgBlocks = coeffsPerDim(dim-1)/_threadsPerBlock;
        int remaining = coeffsPerDim(dim-1)%_threadsPerBlock;

        cudaPitchedPtr dvolumePtr;
        if (dim==3) {
            cudaExtent volExt = make_cudaExtent(coeffsPerDim[0]*sizeof(data_t),coeffsPerDim[1],coeffsPerDim[2]);
            if(cudaMalloc3D(&dvolumePtr,volExt)!=cudaSuccess)
                throw std::bad_alloc();

            //transfer projections
            if(_fast) {
                auto [sinoTex,sino] = copyTextureToGPU<cudaArrayLayered>(y);

                cudaDeviceSynchronize();
                if (numImgBlocks>0) {
                    const dim3 grid(coeffsPerDim(0),coeffsPerDim(1),numImgBlocks);
                    const int threads = _threadsPerBlock;
                    
                    TraverseJosephsCUDA<data_t,3>::traverseAdjointFast(grid,threads,(int8_t*)dvolumePtr.ptr,dvolumePtr.pitch,sinoTex,(int8_t*)_rayOrigins.ptr,_rayOrigins.pitch,(int8_t*)_projMatrices.ptr,_projMatrices.pitch,_rangeDescriptor->getNumberOfCoefficientsPerDimension()(2));    
                }

                if (remaining>0) {
                    cudaStream_t remStream;
                    if(cudaStreamCreate(&remStream)!=cudaSuccess)
                        throw std::logic_error("Couldn't create stream for remaining images");
                    const dim3 grid(coeffsPerDim(0),coeffsPerDim(1),1);
                    const int threads = remaining;
                    int8_t* volPtr = (int8_t*)dvolumePtr.ptr + numImgBlocks*_threadsPerBlock*dvolumePtr.pitch*coeffsPerDim(1);

                    TraverseJosephsCUDA<data_t,3>::traverseAdjointFast(grid,threads,volPtr,dvolumePtr.pitch,sinoTex,(int8_t*)_rayOrigins.ptr,_rayOrigins.pitch,(int8_t*)_projMatrices.ptr,_projMatrices.pitch,_rangeDescriptor->getNumberOfCoefficientsPerDimension()(2),numImgBlocks*_threadsPerBlock,remStream);

                    if (cudaStreamDestroy(remStream)!=cudaSuccess) 
                        Logger::get("JosephsMethodCUDA")->error("Couldn't destroy GPU stream; This may cause problems later.");
                }

                //synchonize because we are using multiple streams
                cudaDeviceSynchronize();

                if(cudaDestroyTextureObject(sinoTex)!=cudaSuccess)
                    Logger::get("JosephsMethodCUDA")->error("Couldn't destroy texture object; This may cause problems later.");

                if(cudaFreeArray(sino)!=cudaSuccess)
                    Logger::get("JosephsMethodCUDA")->error("Couldn't free GPU memory; This may cause problems later.");
            }
            else {
                if(cudaMemset3DAsync(dvolumePtr,0,volExt)!=cudaSuccess)
                    throw std::logic_error("JosephsMethodCUDA::applyAdjoint: Could not zero-initialize volume on GPU.");


                cudaPitchedPtr dsinoPtr;
                IndexVector_t rangeDims = _rangeDescriptor->getNumberOfCoefficientsPerDimension();
                cudaExtent sinoExt = make_cudaExtent(rangeDims[0]*sizeof(data_t),rangeDims[1],rangeDims[2]);
                
                if(cudaMalloc3D(&dsinoPtr,sinoExt)!=cudaSuccess)
                    throw std::bad_alloc();
            
                copy3DDataContainer<cudaMemcpyHostToDevice>((void*)&y[0] ,dsinoPtr,sinoExt);
            
                //perform projection
                int numImgBlocks = rangeDims(2)/_threadsPerBlock;
                int remaining = rangeDims(2)%_threadsPerBlock;

                IndexVector_t bmax = _boundingBox._max.template cast<index_t>();
                typename TraverseJosephsCUDA<data_t,3>::BoundingBox boxMax;
                boxMax.max[0] = bmax[0];
                boxMax.max[1] = bmax[1];
                boxMax.max[2] = bmax[2];
                //synchronize because we are using multiple streams
                cudaDeviceSynchronize();
                if (numImgBlocks>0) {
                    const dim3 grid(rangeDims(0),rangeDims(1),numImgBlocks);
                    const int threads = _threadsPerBlock;
                    TraverseJosephsCUDA<data_t,3>::traverseAdjoint(grid, threads,(int8_t*)dvolumePtr.ptr,dvolumePtr.pitch,(int8_t*) dsinoPtr.ptr, dsinoPtr.pitch,(int8_t*)_rayOrigins.ptr,_rayOrigins.pitch,(int8_t*)_projInvMatrices.ptr,_projInvMatrices.pitch,boxMax);
                }

                if (remaining>0) {
                    cudaStream_t remStream;
                    if(cudaStreamCreate(&remStream)!=cudaSuccess)
                        throw std::logic_error("JosephsMethodCUDA::applyAdjoint: Couldn't create stream for remaining images");
                    
                    const dim3 grid(rangeDims(0),rangeDims(1),1);
                    const int threads = remaining;
                    int8_t* imgPtr = (int8_t*)dsinoPtr.ptr + numImgBlocks*_threadsPerBlock*dsinoPtr.pitch*rangeDims(1);
                    int8_t* rayPtr = (int8_t*)_rayOrigins.ptr + numImgBlocks*_threadsPerBlock*_rayOrigins.pitch;
                    int8_t* matrixPtr = (int8_t*)_projInvMatrices.ptr + numImgBlocks*_threadsPerBlock*_projInvMatrices.pitch*3;
                    TraverseJosephsCUDA<data_t,3>::traverseAdjoint(grid, threads,(int8_t*)dvolumePtr.ptr,dvolumePtr.pitch,imgPtr, dsinoPtr.pitch,rayPtr,_rayOrigins.pitch,matrixPtr,_projInvMatrices.pitch,boxMax,remStream);
                    
                    if (cudaStreamDestroy(remStream)!=cudaSuccess) 
                        Logger::get("JosephsMethodCUDA")->error("Couldn't destroy GPU stream; This may cause problems later.");
                }

                //synchonize because we are using multiple streams
                cudaDeviceSynchronize();

                //free allocated memory
                if (cudaFree(dsinoPtr.ptr) != cudaSuccess)
                    Logger::get("JosephsMethodCUDA")->error("Couldn't free GPU memory; This may cause problems later.");
            }

            //retrieve results from GPU
            copy3DDataContainer<cudaMemcpyDeviceToHost,false>((void*)&Aty[0],dvolumePtr,volExt);
        }
        else {
            if(cudaMallocPitch(&dvolumePtr.ptr,&dvolumePtr.pitch,coeffsPerDim[0]*sizeof(data_t),coeffsPerDim[1])!=cudaSuccess)
                throw std::bad_alloc();

            if(_fast) {
                //transfer projections
                auto [sinoTex,sino] = copyTextureToGPU<cudaArrayLayered>(y);

                cudaDeviceSynchronize();
                if (numImgBlocks>0) {
                    const dim3 grid(coeffsPerDim(0),numImgBlocks);
                    const int threads = _threadsPerBlock;
                    
                    TraverseJosephsCUDA<data_t,2>::traverseAdjointFast(grid,threads,(int8_t*)dvolumePtr.ptr,dvolumePtr.pitch,sinoTex,(int8_t*)_rayOrigins.ptr,_rayOrigins.pitch,(int8_t*)_projMatrices.ptr,_projMatrices.pitch,_rangeDescriptor->getNumberOfCoefficientsPerDimension()(dim-1));    
                }

                if (remaining>0) {
                    cudaStream_t remStream;
                    if(cudaStreamCreate(&remStream)!=cudaSuccess)
                        throw std::logic_error("JosephsMethodCUDA::applyAdjoint: Couldn't create stream for remaining images");
                    const dim3 grid(coeffsPerDim(0),1);
                    const int threads = remaining;
                    int8_t* volPtr = (int8_t*)dvolumePtr.ptr + numImgBlocks*_threadsPerBlock*dvolumePtr.pitch;

                    TraverseJosephsCUDA<data_t,2>::traverseAdjointFast(grid,threads,volPtr,dvolumePtr.pitch,sinoTex,(int8_t*)_rayOrigins.ptr,_rayOrigins.pitch,(int8_t*)_projMatrices.ptr,_projMatrices.pitch,_rangeDescriptor->getNumberOfCoefficientsPerDimension()(dim-1),numImgBlocks*_threadsPerBlock,remStream);

                    if (cudaStreamDestroy(remStream)!=cudaSuccess) 
                        Logger::get("JosephsMethodCUDA")->error("Couldn't destroy GPU stream; This may cause problems later.");
                }

                //synchonize because we are using multiple streams
                cudaDeviceSynchronize();

                if(cudaDestroyTextureObject(sinoTex)!=cudaSuccess)
                    Logger::get("JosephsMethodCUDA")->error("Couldn't destroy texture object; This may cause problems later.");

                if(cudaFreeArray(sino)!=cudaSuccess)
                    Logger::get("JosephsMethodCUDA")->error("Couldn't free GPU memory; This may cause problems later.");
            }
            else {
                if(cudaMemset2DAsync(dvolumePtr.ptr,dvolumePtr.pitch,0,coeffsPerDim[0]*sizeof(data_t),coeffsPerDim[1]) !=cudaSuccess)
                    throw std::logic_error("JosephsMethodCUDA::applyAdjoint: Could not zero-initialize volume on GPU.");

                cudaPitchedPtr dsinoPtr;
                IndexVector_t rangeDims = _rangeDescriptor->getNumberOfCoefficientsPerDimension();
                if(cudaMallocPitch(&dsinoPtr.ptr,&dsinoPtr.pitch,rangeDims[0]*sizeof(data_t),rangeDims[1])!=cudaSuccess)
                    throw std::bad_alloc();
            
                if(cudaMemcpy2DAsync(dsinoPtr.ptr,dsinoPtr.pitch,(void*)&y[0],rangeDims[0]*sizeof(data_t),rangeDims[0]*sizeof(data_t),rangeDims[1],cudaMemcpyHostToDevice)!=cudaSuccess)
                    throw std::logic_error("JosephsMethodCUDA::applyAdjoint: Couldn't transfer sinogram to GPU.");
            
                //perform backprojection
                int numImgBlocks = rangeDims(dim-1)/_threadsPerBlock;
                int remaining = rangeDims(dim-1)%_threadsPerBlock;

                IndexVector_t bmax = _boundingBox._max.template cast<index_t>();
                typename TraverseJosephsCUDA<data_t,2>::BoundingBox boxMax;
                boxMax.max[0] = bmax[0];
                boxMax.max[1] = bmax[1];
                //synchronize because we are using multiple streams
                cudaDeviceSynchronize();
                if (numImgBlocks>0) {
                    const dim3 grid(rangeDims(0),numImgBlocks);
                    const int threads = _threadsPerBlock;
                    TraverseJosephsCUDA<data_t,2>::traverseAdjoint(grid, threads,(int8_t*)dvolumePtr.ptr,dvolumePtr.pitch,(int8_t*) dsinoPtr.ptr, dsinoPtr.pitch,(int8_t*)_rayOrigins.ptr,_rayOrigins.pitch,(int8_t*)_projInvMatrices.ptr,_projInvMatrices.pitch,boxMax);
                }

                if (remaining>0) {
                    cudaStream_t remStream;
                    if(cudaStreamCreate(&remStream)!=cudaSuccess)
                        throw std::logic_error("JosephsMethodCUDA::applyAdjoint: Couldn't create stream for remaining images");
                    
                    const dim3 grid(rangeDims(0),1);
                    const int threads = remaining;
                    int8_t* imgPtr = (int8_t*)dsinoPtr.ptr + numImgBlocks*_threadsPerBlock*dsinoPtr.pitch;
                    int8_t* rayPtr = (int8_t*)_rayOrigins.ptr + numImgBlocks*_threadsPerBlock*_rayOrigins.pitch;
                    int8_t* matrixPtr = (int8_t*)_projInvMatrices.ptr + numImgBlocks*_threadsPerBlock*_projInvMatrices.pitch*dim;
                    TraverseJosephsCUDA<data_t,2>::traverseAdjoint(grid, threads,(int8_t*)dvolumePtr.ptr,dvolumePtr.pitch,imgPtr, dsinoPtr.pitch,rayPtr,_rayOrigins.pitch,matrixPtr,_projInvMatrices.pitch,boxMax,remStream);
                    
                    if (cudaStreamDestroy(remStream)!=cudaSuccess) 
                        Logger::get("JosephsMethodCUDA")->error("Couldn't destroy GPU stream; This may cause problems later.");
                }

                //synchonize because we are using multiple streams
                cudaDeviceSynchronize();

                //free allocated memory
                if (cudaFree(dsinoPtr.ptr) != cudaSuccess)
                    Logger::get("JosephsMethodCUDA")->error("Couldn't free GPU memory; This may cause problems later.");
            }

            //retrieve results from GPU
            if (cudaMemcpy2D((void*)&Aty[0],coeffsPerDim[0]*sizeof(data_t),dvolumePtr.ptr,dvolumePtr.pitch,sizeof(data_t)*coeffsPerDim[0],coeffsPerDim[1],cudaMemcpyDeviceToHost)!=cudaSuccess)
                throw std::logic_error("JosephsMethodCUDA::applyAdjoint: Couldn't retrieve results from GPU");
        }

        //free allocated memory
        if (cudaFree(dvolumePtr.ptr) != cudaSuccess)
            Logger::get("JosephsMethodCUDA")->error("Couldn't free GPU memory; This may cause problems later.");
    }

    template <typename data_t>
    template <cudaMemcpyKind direction, bool async>
    void JosephsMethodCUDA<data_t>::copy3DDataContainer(void* hostData,const cudaPitchedPtr& gpuData, const cudaExtent& extent) const
    {
        cudaMemcpy3DParms cpyParams = {0};
        cpyParams.extent = extent;
        cpyParams.kind = direction;

        cudaPitchedPtr tmp = {0};
        tmp.ptr = hostData;
        tmp.pitch = extent.width;
        tmp.xsize = extent.width;
        tmp.ysize = extent.height;

        if (direction==cudaMemcpyHostToDevice) {
            cpyParams.dstPtr = gpuData;
            cpyParams.srcPtr = tmp;
        }
        else if(direction==cudaMemcpyDeviceToHost){
            cpyParams.srcPtr = gpuData;
            cpyParams.dstPtr = tmp;
        }
        else {
            throw std::logic_error("Can only copy data between device and host");
        }

        if (async) {
            if(cudaMemcpy3DAsync(&cpyParams)!=cudaSuccess)
                throw std::logic_error("Failed copying data between device and host");
        }
        else {
            if(cudaMemcpy3D(&cpyParams)!=cudaSuccess)
                throw std::logic_error("Failed copying data between device and host");
        }
            
        
    }

    template <typename data_t>
    template <unsigned int flags> 
    std::pair<cudaTextureObject_t, cudaArray*> JosephsMethodCUDA<data_t>::copyTextureToGPU(const DataContainer<data_t>& hostData) const
    {
        //transfer volume as texture
        auto coeffsPerDim =  hostData.getDataDescriptor().getNumberOfCoefficientsPerDimension();

        cudaArray* volume;
        cudaTextureObject_t volumeTex = 0;

        cudaChannelFormatDesc channelDesc;

        if constexpr(sizeof(data_t) == 4)
            channelDesc = cudaCreateChannelDesc(sizeof(data_t)*8,0,0,0,cudaChannelFormatKindFloat);
        else if(sizeof(data_t) == 8) 
            channelDesc = cudaCreateChannelDesc(32,32,0,0,cudaChannelFormatKindSigned);
        else
            throw std::invalid_argument("JosephsMethodCUDA::copyTextureToGPU: only supports DataContainer<data_t> with data_t of length 4 or 8 bytes");

        if (hostData.getDataDescriptor().getNumberOfDimensions() == 3) {
            cudaExtent volumeExtent = make_cudaExtent(coeffsPerDim[0],coeffsPerDim[1],coeffsPerDim[2]);
            if(cudaMalloc3DArray(&volume,&channelDesc,volumeExtent,flags)!=cudaSuccess)
                throw std::bad_alloc();
            cudaMemcpy3DParms cpyParams = {0};
            cpyParams.srcPtr.ptr = (void*)&hostData[0];
            cpyParams.srcPtr.pitch = coeffsPerDim[0]*sizeof(data_t);
            cpyParams.srcPtr.xsize = coeffsPerDim[0]*sizeof(data_t);
            cpyParams.srcPtr.ysize = coeffsPerDim[1];
            cpyParams.dstArray = volume;
            cpyParams.extent = volumeExtent;
            cpyParams.kind = cudaMemcpyHostToDevice;

            if (cudaMemcpy3DAsync(&cpyParams)!=cudaSuccess)
                throw std::logic_error("JosephsMethodCUDA::copyTextureToGPU: Could not transfer data to GPU.");
        }
        else {
            //CUDA has a very weird way of handling layered 1D arrays
            if (flags==cudaArrayLayered) {

                //must be allocated as a 3D Array of height 0
                cudaExtent volumeExtent = make_cudaExtent(coeffsPerDim[0],0,coeffsPerDim[1]);
                if(cudaMalloc3DArray(&volume,&channelDesc,volumeExtent,flags)!=cudaSuccess) 
                    throw std::bad_alloc();
                
                //adjust height to 1 for copy
                volumeExtent.height=1;
                cudaMemcpy3DParms cpyParams = {0};
                cpyParams.srcPos = make_cudaPos(0,0,0);
                cpyParams.dstPos = make_cudaPos(0,0,0);
                cpyParams.srcPtr = make_cudaPitchedPtr((void*)&hostData[0],coeffsPerDim[0]*sizeof(data_t),coeffsPerDim[0],1);
                cpyParams.dstArray = volume;
                cpyParams.extent = volumeExtent;
                cpyParams.kind = cudaMemcpyHostToDevice;

                if (cudaMemcpy3DAsync(&cpyParams)!=cudaSuccess)
                    throw std::logic_error("JosephsMethodCUDA::copyTextureToGPU: Could not transfer data to GPU.");
            }
            else {
                if(cudaMallocArray(&volume,&channelDesc,coeffsPerDim[0],coeffsPerDim[1],flags)!=cudaSuccess)
                    throw std::bad_alloc();
                
                if(cudaMemcpy2DToArrayAsync(volume,0,0,&hostData[0],
                                        coeffsPerDim[0]*sizeof(data_t),
                                        coeffsPerDim[0]*sizeof(data_t),
                                        coeffsPerDim[1],
                                        cudaMemcpyHostToDevice)!=cudaSuccess)
                    throw std::logic_error("JosephsMethodCUDA::copyTextureToGPU: Could not transfer data to GPU.");
            }
        }
        
        

        cudaResourceDesc resDesc;
        std::memset(&resDesc,0,sizeof(resDesc));
        resDesc.resType=cudaResourceTypeArray;
        resDesc.res.array.array = volume;

        cudaTextureDesc texDesc;
        std::memset(&texDesc,0,sizeof(texDesc));
        texDesc.addressMode[0] = flags ? cudaAddressModeBorder:cudaAddressModeClamp;
        texDesc.addressMode[1] = flags ? cudaAddressModeBorder:cudaAddressModeClamp;
        texDesc.addressMode[2] = flags ? cudaAddressModeBorder:cudaAddressModeClamp;
        texDesc.filterMode = sizeof(data_t)==4 ? cudaFilterModeLinear : cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        if(cudaCreateTextureObject(&volumeTex,&resDesc,&texDesc,NULL)!=cudaSuccess)
            throw std::logic_error("Couldn't create texture object");
        
        return std::pair<cudaTextureObject_t,cudaArray*>(volumeTex,volume);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class JosephsMethodCUDA<float>;
    template class JosephsMethodCUDA<double>;
}