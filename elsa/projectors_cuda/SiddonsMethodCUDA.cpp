#include "SiddonsMethodCUDA.h"
#include "LogGuard.h"
#include "Timer.h"

#include "Logger.h"
namespace elsa
{
    template <typename data_t>
    SiddonsMethodCUDA<data_t>::SiddonsMethodCUDA(const DataDescriptor& domainDescriptor,
                                                 const DataDescriptor& rangeDescriptor,
                                                 const std::vector<Geometry>& geometryList)
        : LinearOperator<data_t>(domainDescriptor, rangeDescriptor),
          _geometryList(geometryList),
          _boundingBox(_domainDescriptor->getNumberOfCoefficientsPerDimension())
    {
        auto dim = _domainDescriptor->getNumberOfDimensions();
        if (dim != _rangeDescriptor->getNumberOfDimensions()) {
            throw std::logic_error(
                std::string("SiddonsMethodCUDA: domain and range dimension need to match"));
        }

        if (dim != 2 && dim != 3) {
            throw std::logic_error("SiddonsMethodCUDA: only supporting 2d/3d operations");
        }

        if (geometryList.empty()) {
            throw std::logic_error("SiddonsMethodCUDA: geometry list was empty");
        }

        // allocate device memory and copy ray origins and the inverse of the significant part of
        // projection matrices to device
        cudaExtent extent = make_cudaExtent(dim * sizeof(real_t), dim, geometryList.size());

        if (cudaMallocPitch(&_rayOrigins.ptr, &_rayOrigins.pitch, dim * sizeof(real_t),
                            geometryList.size())
            != cudaSuccess)
            throw std::bad_alloc();
        _rayOrigins.xsize = dim;
        _rayOrigins.ysize = geometryList.size();

        if (cudaMalloc3D(&_projInvMatrices, extent) != cudaSuccess)
            throw std::bad_alloc();

        int8_t* projPtr = (int8_t*) _projInvMatrices.ptr;
        index_t projPitch = _projInvMatrices.pitch;
        int8_t* rayBasePtr = (int8_t*) _rayOrigins.ptr;
        index_t rayPitch = _rayOrigins.pitch;
        for (index_t i = 0; i < geometryList.size(); i++) {
            RealMatrix_t P = geometryList[i].getInverseProjectionMatrix().block(0, 0, dim, dim);
            int8_t* slice = projPtr + i * projPitch * dim;
            // CUDA also uses a column-major representation, directly transfer matrix
            // transfer inverse of projection matrix
            if (cudaMemcpy2DAsync(slice, projPitch, P.data(), dim * sizeof(real_t),
                                  dim * sizeof(real_t), dim, cudaMemcpyHostToDevice)
                != cudaSuccess)
                throw std::logic_error(
                    "SiddonsMethodCUDA: Could not transfer inverse projection matrices to GPU.");

            int8_t* rayPtr = rayBasePtr + i * rayPitch;
            // get camera center using direct inverse
            RealVector_t ro = -P * geometryList[i].getProjectionMatrix().block(0, dim, dim, 1);
            // transfer ray origin
            if (cudaMemcpyAsync(rayPtr, ro.data(), dim * sizeof(real_t), cudaMemcpyHostToDevice)
                != cudaSuccess)
                throw std::logic_error("SiddonsMethodCUDA: Could not transfer ray origins to GPU.");
        }
    }

    template <typename data_t>
    SiddonsMethodCUDA<data_t>::~SiddonsMethodCUDA()
    {
        // Free CUDA resources
        if (cudaFree(_rayOrigins.ptr) != cudaSuccess
            || cudaFree(_projInvMatrices.ptr) != cudaSuccess)
            Logger::get("SiddonsMethodCUDA")
                ->error("Couldn't free GPU memory; This may cause problems later.");
    }

    template <typename data_t>
    SiddonsMethodCUDA<data_t>* SiddonsMethodCUDA<data_t>::cloneImpl() const
    {
        return new SiddonsMethodCUDA<data_t>(*_domainDescriptor, *_rangeDescriptor, _geometryList);
    }

    template <typename data_t>
    void SiddonsMethodCUDA<data_t>::applyImpl(const DataContainer<data_t>& x,
                                              DataContainer<data_t>& Ax) const
    {
        Timer<> timeGuard("SiddonsMethodCUDA", "apply");

        traverseVolume<false>((void*) &(x[0]), (void*) &(Ax[0]));
    }

    template <typename data_t>
    void SiddonsMethodCUDA<data_t>::applyAdjointImpl(const DataContainer<data_t>& y,
                                                     DataContainer<data_t>& Aty) const
    {
        Timer<> timeguard("SiddonsMethodCUDA", "applyAdjoint");

        traverseVolume<true>((void*) &(Aty[0]), (void*) &(y[0]));
    }

    template <typename data_t>
    bool SiddonsMethodCUDA<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        auto otherSM = dynamic_cast<const SiddonsMethodCUDA*>(&other);
        if (!otherSM)
            return false;

        if (_geometryList != otherSM->_geometryList)
            return false;

        return true;
    }

    template <typename data_t>
    template <bool adjoint>
    void SiddonsMethodCUDA<data_t>::traverseVolume(void* volumePtr, void* sinoPtr) const
    {
        auto coeffsPerDim = _domainDescriptor->getNumberOfCoefficientsPerDimension();
        IndexVector_t rangeDims = _rangeDescriptor->getNumberOfCoefficientsPerDimension();
        cudaPitchedPtr dvolumePtr;
        cudaPitchedPtr dsinoPtr;

        int numDim = _domainDescriptor->getNumberOfDimensions();
        int numImgBlocks = rangeDims(numDim - 1) / _threadsPerBlock;
        int remaining = rangeDims(numDim - 1) % _threadsPerBlock;
        if (numDim == 3) {
            typename TraverseSiddonsCUDA<data_t, 3>::BoundingBox boxMax;
            boxMax.max[0] = _boundingBox._max.template cast<uint32_t>()[0];
            boxMax.max[1] = _boundingBox._max.template cast<uint32_t>()[1];
            boxMax.max[2] = _boundingBox._max.template cast<uint32_t>()[2];

            // transfer volume and sinogram
            cudaExtent volExt =
                make_cudaExtent(coeffsPerDim[0] * sizeof(data_t), coeffsPerDim[1], coeffsPerDim[2]);
            if (cudaMalloc3D(&dvolumePtr, volExt) != cudaSuccess)
                throw std::bad_alloc();

            cudaExtent sinoExt =
                make_cudaExtent(rangeDims[0] * sizeof(data_t), rangeDims[1], rangeDims[2]);
            if (cudaMalloc3D(&dsinoPtr, sinoExt) != cudaSuccess)
                throw std::bad_alloc();

            if (adjoint) {
                copy3DDataContainerGPU<cudaMemcpyHostToDevice>(sinoPtr, dsinoPtr, sinoExt);
                if (cudaMemset3DAsync(dvolumePtr, 0, volExt) != cudaSuccess)
                    throw std::logic_error("SiddonsMethodCUDA::traverseVolume: Could not "
                                           "zero-initialize volume on GPU.");
            } else {
                copy3DDataContainerGPU<cudaMemcpyHostToDevice>(volumePtr, dvolumePtr, volExt);
                if (cudaMemset3DAsync(dsinoPtr, 0, sinoExt) != cudaSuccess)
                    throw std::logic_error("SiddonsMethodCUDA::traverseVolume: Could not "
                                           "zero-initialize sinogram on GPU.");
            }

            // synchronize because we are using multiple streams
            cudaDeviceSynchronize();
            if (numImgBlocks > 0) {
                const dim3 grid(rangeDims(0), rangeDims(1), numImgBlocks);
                const int threads = _threadsPerBlock;
                if (adjoint)
                    TraverseSiddonsCUDA<data_t, 3>::traverseAdjoint(
                        grid, threads, (int8_t*) dvolumePtr.ptr, dvolumePtr.pitch,
                        (int8_t*) dsinoPtr.ptr, dsinoPtr.pitch, (int8_t*) _rayOrigins.ptr,
                        _rayOrigins.pitch, (int8_t*) _projInvMatrices.ptr, _projInvMatrices.pitch,
                        boxMax);
                else
                    TraverseSiddonsCUDA<data_t, 3>::traverseForward(
                        grid, threads, (int8_t*) dvolumePtr.ptr, dvolumePtr.pitch,
                        (int8_t*) dsinoPtr.ptr, dsinoPtr.pitch, (int8_t*) _rayOrigins.ptr,
                        _rayOrigins.pitch, (int8_t*) _projInvMatrices.ptr, _projInvMatrices.pitch,
                        boxMax);
            }

            if (remaining > 0) {
                cudaStream_t remStream;
                if (cudaStreamCreate(&remStream) != cudaSuccess)
                    throw std::logic_error("Couldn't create stream for remaining images");
                const dim3 grid(rangeDims(0), rangeDims(1), 1);
                const int threads = remaining;
                int8_t* imgPtr = (int8_t*) dsinoPtr.ptr
                                 + numImgBlocks * _threadsPerBlock * dsinoPtr.pitch * rangeDims(1);
                int8_t* rayPtr =
                    (int8_t*) _rayOrigins.ptr + numImgBlocks * _threadsPerBlock * _rayOrigins.pitch;
                int8_t* matrixPtr = (int8_t*) _projInvMatrices.ptr
                                    + numImgBlocks * _threadsPerBlock * _projInvMatrices.pitch * 3;

                if (adjoint)
                    TraverseSiddonsCUDA<data_t, 3>::traverseAdjoint(
                        grid, threads, (int8_t*) dvolumePtr.ptr, dvolumePtr.pitch, imgPtr,
                        dsinoPtr.pitch, rayPtr, _rayOrigins.pitch, matrixPtr,
                        _projInvMatrices.pitch, boxMax, remStream);
                else
                    TraverseSiddonsCUDA<data_t, 3>::traverseForward(
                        grid, threads, (int8_t*) dvolumePtr.ptr, dvolumePtr.pitch, imgPtr,
                        dsinoPtr.pitch, rayPtr, _rayOrigins.pitch, matrixPtr,
                        _projInvMatrices.pitch, boxMax, remStream);

                if (cudaStreamDestroy(remStream) != cudaSuccess)
                    Logger::get("SiddonsMethodCUDA")
                        ->error("Couldn't destroy GPU stream; This may cause problems later.");
            }

            // synchonize because we are using multiple streams
            cudaDeviceSynchronize();
            // retrieve results from GPU
            if (adjoint)
                copy3DDataContainerGPU<cudaMemcpyDeviceToHost, false>(volumePtr, dvolumePtr,
                                                                      volExt);
            else
                copy3DDataContainerGPU<cudaMemcpyDeviceToHost, false>(sinoPtr, dsinoPtr, sinoExt);

        } else {
            typename TraverseSiddonsCUDA<data_t, 2>::BoundingBox boxMax;
            boxMax.max[0] = _boundingBox._max.template cast<uint32_t>()[0];
            boxMax.max[1] = _boundingBox._max.template cast<uint32_t>()[1];

            // transfer volume and sinogram

            if (cudaMallocPitch(&dvolumePtr.ptr, &dvolumePtr.pitch,
                                coeffsPerDim[0] * sizeof(data_t), coeffsPerDim[1])
                != cudaSuccess)
                throw std::bad_alloc();

            if (cudaMallocPitch(&dsinoPtr.ptr, &dsinoPtr.pitch, rangeDims[0] * sizeof(data_t),
                                rangeDims[1])
                != cudaSuccess)
                throw std::bad_alloc();

            if (adjoint) {
                if (cudaMemcpy2DAsync(dsinoPtr.ptr, dsinoPtr.pitch, sinoPtr,
                                      rangeDims[0] * sizeof(data_t), rangeDims[0] * sizeof(data_t),
                                      rangeDims[1], cudaMemcpyHostToDevice)
                    != cudaSuccess)
                    throw std::logic_error(
                        "SiddonsMethodCUDA::traverseVolume: Couldn't transfer sinogram to GPU.");

                if (cudaMemset2DAsync(dvolumePtr.ptr, dvolumePtr.pitch, 0,
                                      coeffsPerDim[0] * sizeof(data_t), coeffsPerDim[1])
                    != cudaSuccess)
                    throw std::logic_error("SiddonsMethodCUDA::traverseVolume: Could not "
                                           "zero-initialize volume on GPU.");
            } else {
                if (cudaMemcpy2DAsync(dvolumePtr.ptr, dvolumePtr.pitch, volumePtr,
                                      coeffsPerDim[0] * sizeof(data_t),
                                      coeffsPerDim[0] * sizeof(data_t), coeffsPerDim[1],
                                      cudaMemcpyHostToDevice)
                    != cudaSuccess)
                    throw std::logic_error(
                        "SiddonsMethodCUDA::traverseVolume: Couldn't transfer volume to GPU.");

                if (cudaMemset2DAsync(dsinoPtr.ptr, dsinoPtr.pitch, 0,
                                      rangeDims[0] * sizeof(data_t), rangeDims[1])
                    != cudaSuccess)
                    throw std::logic_error("SiddonsMethodCUDA::traverseVolume: Could not "
                                           "zero-initialize sinogram on GPU.");
            }

            // perform projection

            // synchronize because we are using multiple streams
            cudaDeviceSynchronize();
            if (numImgBlocks > 0) {
                const dim3 grid(rangeDims(0), numImgBlocks);
                const int threads = _threadsPerBlock;
                if (adjoint)
                    TraverseSiddonsCUDA<data_t, 2>::traverseAdjoint(
                        grid, threads, (int8_t*) dvolumePtr.ptr, dvolumePtr.pitch,
                        (int8_t*) dsinoPtr.ptr, dsinoPtr.pitch, (int8_t*) _rayOrigins.ptr,
                        _rayOrigins.pitch, (int8_t*) _projInvMatrices.ptr, _projInvMatrices.pitch,
                        boxMax);
                else
                    TraverseSiddonsCUDA<data_t, 2>::traverseForward(
                        grid, threads, (int8_t*) dvolumePtr.ptr, dvolumePtr.pitch,
                        (int8_t*) dsinoPtr.ptr, dsinoPtr.pitch, (int8_t*) _rayOrigins.ptr,
                        _rayOrigins.pitch, (int8_t*) _projInvMatrices.ptr, _projInvMatrices.pitch,
                        boxMax);
            }

            if (remaining > 0) {
                cudaStream_t remStream;
                if (cudaStreamCreate(&remStream) != cudaSuccess)
                    throw std::logic_error("Couldn't create stream for remaining images");
                const dim3 grid(rangeDims(0), 1);
                const int threads = remaining;
                int8_t* imgPtr =
                    (int8_t*) dsinoPtr.ptr + numImgBlocks * _threadsPerBlock * dsinoPtr.pitch;
                int8_t* rayPtr =
                    (int8_t*) _rayOrigins.ptr + numImgBlocks * _threadsPerBlock * _rayOrigins.pitch;
                int8_t* matrixPtr =
                    (int8_t*) _projInvMatrices.ptr
                    + numImgBlocks * _threadsPerBlock * _projInvMatrices.pitch * numDim;

                if (adjoint)
                    TraverseSiddonsCUDA<data_t, 2>::traverseAdjoint(
                        grid, threads, (int8_t*) dvolumePtr.ptr, dvolumePtr.pitch, imgPtr,
                        dsinoPtr.pitch, rayPtr, _rayOrigins.pitch, matrixPtr,
                        _projInvMatrices.pitch, boxMax, remStream);
                else
                    TraverseSiddonsCUDA<data_t, 2>::traverseForward(
                        grid, threads, (int8_t*) dvolumePtr.ptr, dvolumePtr.pitch, imgPtr,
                        dsinoPtr.pitch, rayPtr, _rayOrigins.pitch, matrixPtr,
                        _projInvMatrices.pitch, boxMax, remStream);
                if (cudaStreamDestroy(remStream) != cudaSuccess)
                    Logger::get("SiddonsMethodCUDA")
                        ->error("Couldn't destroy GPU stream; This may cause problems later.");
            }

            // synchonize because we are using multiple streams
            cudaDeviceSynchronize();
            // retrieve results from GPU
            if (adjoint) {
                if (cudaMemcpy2D(volumePtr, coeffsPerDim[0] * sizeof(data_t), dvolumePtr.ptr,
                                 dvolumePtr.pitch, coeffsPerDim[0] * sizeof(data_t),
                                 coeffsPerDim[1], cudaMemcpyDeviceToHost)
                    != cudaSuccess)
                    throw std::logic_error(
                        "SiddonsMethodCUDA::traverseVolume: Couldn't retrieve results from GPU.");
            } else {
                if (cudaMemcpy2D(sinoPtr, rangeDims[0] * sizeof(data_t), dsinoPtr.ptr,
                                 dsinoPtr.pitch, rangeDims[0] * sizeof(data_t), rangeDims[1],
                                 cudaMemcpyDeviceToHost)
                    != cudaSuccess)
                    throw std::logic_error(
                        "SiddonsMethodCUDA::traverseVolume: Couldn't retrieve results from GPU");
            }
        }

        // free allocated memory
        if (cudaFree(dvolumePtr.ptr) != cudaSuccess || cudaFree(dsinoPtr.ptr) != cudaSuccess)
            Logger::get("SiddonsMethodCUDA")
                ->error("Couldn't free GPU memory; This may cause problems later.");
    }

    template <typename data_t>
    template <cudaMemcpyKind direction, bool async>
    void SiddonsMethodCUDA<data_t>::copy3DDataContainerGPU(void* hostData,
                                                           const cudaPitchedPtr& gpuData,
                                                           const cudaExtent& extent) const
    {
        cudaMemcpy3DParms cpyParams = {0};
        cpyParams.extent = extent;
        cpyParams.kind = direction;

        cudaPitchedPtr tmp = {0};
        tmp.ptr = hostData;
        tmp.pitch = extent.width;
        tmp.xsize = extent.width;
        tmp.ysize = extent.height;

        if (direction == cudaMemcpyHostToDevice) {
            cpyParams.dstPtr = gpuData;
            cpyParams.srcPtr = tmp;
        } else if (direction == cudaMemcpyDeviceToHost) {
            cpyParams.srcPtr = gpuData;
            cpyParams.dstPtr = tmp;
        } else {
            throw std::logic_error("Can only copy data between device and host");
        }

        if (async) {
            if (cudaMemcpy3DAsync(&cpyParams) != cudaSuccess)
                throw std::logic_error("Failed copying data between device and host");
        } else {
            if (cudaMemcpy3D(&cpyParams) != cudaSuccess)
                throw std::logic_error("Failed copying data between device and host");
        }
    }

    // ------------------------------------------
    // explicit template instantiation
    template class SiddonsMethodCUDA<float>;
    template class SiddonsMethodCUDA<double>;
} // namespace elsa