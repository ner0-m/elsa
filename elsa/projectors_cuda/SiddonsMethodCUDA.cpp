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
          _boundingBox(_domainDescriptor->getNumberOfCoefficientsPerDimension()),
          _geometryList(geometryList)
    {
        auto dim = static_cast<std::size_t>(_domainDescriptor->getNumberOfDimensions());
        if (dim != static_cast<std::size_t>(_rangeDescriptor->getNumberOfDimensions())) {
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

        auto* projPtr = (int8_t*) _projInvMatrices.ptr;
        auto projPitch = _projInvMatrices.pitch;
        auto* rayBasePtr = (int8_t*) _rayOrigins.ptr;
        auto rayPitch = _rayOrigins.pitch;
        for (std::size_t i = 0; i < geometryList.size(); i++) {
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
            RealVector_t ro =
                -P
                * geometryList[i].getProjectionMatrix().block(0, static_cast<index_t>(dim), dim, 1);
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
        auto domainDims = _domainDescriptor->getNumberOfCoefficientsPerDimension();
        auto domainDimsui = domainDims.template cast<unsigned int>();
        IndexVector_t rangeDims = _rangeDescriptor->getNumberOfCoefficientsPerDimension();
        auto rangeDimsui = rangeDims.template cast<unsigned int>();
        cudaPitchedPtr dvolumePtr;
        cudaPitchedPtr dsinoPtr;

        if (_domainDescriptor->getNumberOfDimensions() == 3) {
            typename TraverseSiddonsCUDA<data_t, 3>::BoundingBox boundingBox;
            boundingBox._max[0] = _boundingBox._max.template cast<uint32_t>()[0];
            boundingBox._max[1] = _boundingBox._max.template cast<uint32_t>()[1];
            boundingBox._max[2] = _boundingBox._max.template cast<uint32_t>()[2];

            // transfer volume and sinogram
            cudaExtent volExt =
                make_cudaExtent(domainDimsui[0] * sizeof(data_t), domainDimsui[1], domainDimsui[2]);
            if (cudaMalloc3D(&dvolumePtr, volExt) != cudaSuccess)
                throw std::bad_alloc();

            cudaExtent sinoExt =
                make_cudaExtent(rangeDimsui[0] * sizeof(data_t), rangeDimsui[1], rangeDimsui[2]);
            if (cudaMalloc3D(&dsinoPtr, sinoExt) != cudaSuccess)
                throw std::bad_alloc();

            if (adjoint) {
                copy3DDataContainerGPU<cudaMemcpyHostToDevice>(sinoPtr, dsinoPtr, sinoExt);
                if (cudaMemset3DAsync(dvolumePtr, 0, volExt) != cudaSuccess)
                    throw std::logic_error("SiddonsMethodCUDA::traverseVolume: Could not "
                                           "zero-initialize volume on GPU.");
            } else {
                copy3DDataContainerGPU<cudaMemcpyHostToDevice>(volumePtr, dvolumePtr, volExt);
            }

            dim3 sinogramDims(rangeDimsui[2], rangeDimsui[1], rangeDimsui[0]);
            // synchronize because we are using multiple streams
            cudaDeviceSynchronize();
            if (adjoint) {
                TraverseSiddonsCUDA<data_t, 3>::traverseAdjoint(
                    sinogramDims, THREADS_PER_BLOCK, (int8_t*) dvolumePtr.ptr, dvolumePtr.pitch,
                    (int8_t*) dsinoPtr.ptr, dsinoPtr.pitch, (int8_t*) _rayOrigins.ptr,
                    static_cast<uint32_t>(_rayOrigins.pitch), (int8_t*) _projInvMatrices.ptr,
                    static_cast<uint32_t>(_projInvMatrices.pitch), boundingBox);
            } else {
                TraverseSiddonsCUDA<data_t, 3>::traverseForward(
                    sinogramDims, THREADS_PER_BLOCK, (int8_t*) dvolumePtr.ptr, dvolumePtr.pitch,
                    (int8_t*) dsinoPtr.ptr, dsinoPtr.pitch, (int8_t*) _rayOrigins.ptr,
                    static_cast<uint32_t>(_rayOrigins.pitch), (int8_t*) _projInvMatrices.ptr,
                    static_cast<uint32_t>(_projInvMatrices.pitch), boundingBox);
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
            typename TraverseSiddonsCUDA<data_t, 2>::BoundingBox boundingBox;
            boundingBox._max[0] = _boundingBox._max.template cast<uint32_t>()[0];
            boundingBox._max[1] = _boundingBox._max.template cast<uint32_t>()[1];

            // transfer volume and sinogram

            if (cudaMallocPitch(&dvolumePtr.ptr, &dvolumePtr.pitch,
                                domainDimsui[0] * sizeof(data_t), domainDimsui[1])
                != cudaSuccess)
                throw std::bad_alloc();

            if (cudaMallocPitch(&dsinoPtr.ptr, &dsinoPtr.pitch, rangeDimsui[0] * sizeof(data_t),
                                rangeDimsui[1])
                != cudaSuccess)
                throw std::bad_alloc();

            if (adjoint) {
                if (cudaMemcpy2DAsync(
                        dsinoPtr.ptr, dsinoPtr.pitch, sinoPtr, rangeDimsui[0] * sizeof(data_t),
                        rangeDimsui[0] * sizeof(data_t), rangeDimsui[1], cudaMemcpyHostToDevice)
                    != cudaSuccess)
                    throw std::logic_error(
                        "SiddonsMethodCUDA::traverseVolume: Couldn't transfer sinogram to GPU.");

                if (cudaMemset2DAsync(dvolumePtr.ptr, dvolumePtr.pitch, 0,
                                      domainDimsui[0] * sizeof(data_t), domainDimsui[1])
                    != cudaSuccess)
                    throw std::logic_error("SiddonsMethodCUDA::traverseVolume: Could not "
                                           "zero-initialize volume on GPU.");
            } else {
                if (cudaMemcpy2DAsync(dvolumePtr.ptr, dvolumePtr.pitch, volumePtr,
                                      domainDimsui[0] * sizeof(data_t),
                                      domainDimsui[0] * sizeof(data_t), domainDimsui[1],
                                      cudaMemcpyHostToDevice)
                    != cudaSuccess)
                    throw std::logic_error(
                        "SiddonsMethodCUDA::traverseVolume: Couldn't transfer volume to GPU.");
            }

            // perform projection
            dim3 sinogramDims(rangeDimsui[1], 1, rangeDimsui[0]);
            // synchronize because we are using multiple streams
            cudaDeviceSynchronize();
            if (adjoint) {
                TraverseSiddonsCUDA<data_t, 2>::traverseAdjoint(
                    sinogramDims, THREADS_PER_BLOCK, (int8_t*) dvolumePtr.ptr, dvolumePtr.pitch,
                    (int8_t*) dsinoPtr.ptr, dsinoPtr.pitch, (int8_t*) _rayOrigins.ptr,
                    static_cast<uint32_t>(_rayOrigins.pitch), (int8_t*) _projInvMatrices.ptr,
                    static_cast<uint32_t>(_projInvMatrices.pitch), boundingBox);
            } else {
                TraverseSiddonsCUDA<data_t, 2>::traverseForward(
                    sinogramDims, THREADS_PER_BLOCK, (int8_t*) dvolumePtr.ptr, dvolumePtr.pitch,
                    (int8_t*) dsinoPtr.ptr, dsinoPtr.pitch, (int8_t*) _rayOrigins.ptr,
                    static_cast<uint32_t>(_rayOrigins.pitch), (int8_t*) _projInvMatrices.ptr,
                    static_cast<uint32_t>(_projInvMatrices.pitch), boundingBox);
            }
            // synchonize because we are using multiple streams
            cudaDeviceSynchronize();

            // retrieve results from GPU
            if (adjoint) {
                if (cudaMemcpy2D(volumePtr, domainDimsui[0] * sizeof(data_t), dvolumePtr.ptr,
                                 dvolumePtr.pitch, domainDimsui[0] * sizeof(data_t),
                                 domainDimsui[1], cudaMemcpyDeviceToHost)
                    != cudaSuccess)
                    throw std::logic_error(
                        "SiddonsMethodCUDA::traverseVolume: Couldn't retrieve results from GPU.");
            } else {
                if (cudaMemcpy2D(sinoPtr, rangeDimsui[0] * sizeof(data_t), dsinoPtr.ptr,
                                 dsinoPtr.pitch, rangeDimsui[0] * sizeof(data_t), rangeDimsui[1],
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
        cudaMemcpy3DParms cpyParams = {};
        cpyParams.extent = extent;
        cpyParams.kind = direction;

        cudaPitchedPtr tmp =
            make_cudaPitchedPtr(hostData, extent.width, extent.width, extent.height);

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