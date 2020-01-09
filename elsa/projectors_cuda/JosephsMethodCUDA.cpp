#include "JosephsMethodCUDA.h"
#include "LogGuard.h"
#include "Timer.h"

namespace elsa
{
    template <typename data_t>
    JosephsMethodCUDA<data_t>::JosephsMethodCUDA(const DataDescriptor& domainDescriptor,
                                                 const DataDescriptor& rangeDescriptor,
                                                 const std::vector<Geometry>& geometryList,
                                                 bool fast)
        : LinearOperator<data_t>(domainDescriptor, rangeDescriptor),
          _boundingBox{_domainDescriptor->getNumberOfCoefficientsPerDimension()},
          _geometryList{geometryList},
          _fast{fast}
    {
        auto dim = static_cast<std::size_t>(_domainDescriptor->getNumberOfDimensions());
        if (dim != static_cast<std::size_t>(_rangeDescriptor->getNumberOfDimensions())) {
            throw std::logic_error(
                std::string("JosephsMethodCUDA: domain and range dimension need to match"));
        }

        if (dim != 2 && dim != 3) {
            throw std::logic_error("JosephsMethodCUDA: only supporting 2d/3d operations");
        }

        if (geometryList.empty()) {
            throw std::logic_error("JosephsMethodCUDA: geometry list was empty");
        }

        // allocate device memory and copy ray origins and the inverse of projection matrices to
        // device
        cudaExtent extent = make_cudaExtent(dim * sizeof(real_t), dim, geometryList.size());

        if (cudaMallocPitch(&_rayOrigins.ptr, &_rayOrigins.pitch, dim * sizeof(real_t),
                            geometryList.size())
            != cudaSuccess)
            throw std::bad_alloc();

        _rayOrigins.xsize = dim;
        _rayOrigins.ysize = geometryList.size();

        if (cudaMalloc3D(&_projInvMatrices, extent) != cudaSuccess)
            throw std::bad_alloc();

        if (_fast && cudaMalloc3D(&_projMatrices, extent) != cudaSuccess)
            throw std::bad_alloc();

        auto projPitch = _projInvMatrices.pitch;
        auto* rayBasePtr = (int8_t*) _rayOrigins.ptr;
        auto rayPitch = _rayOrigins.pitch;
        for (std::size_t i = 0; i < geometryList.size(); i++) {
            RealMatrix_t P = geometryList[i].getInverseProjectionMatrix().block(0, 0, dim, dim);
            auto* slice = (int8_t*) _projInvMatrices.ptr + i * projPitch * dim;

            // transfer inverse of projection matrix
            if (cudaMemcpy2D(slice, projPitch, P.data(), dim * sizeof(real_t), dim * sizeof(real_t),
                             dim, cudaMemcpyHostToDevice)
                != cudaSuccess)
                throw std::logic_error(
                    "JosephsMethodCUDA: Could not transfer inverse projection matrices to GPU.");

            // transfer projection matrix if _fast flag is set
            if (_fast) {
                P = geometryList[i].getProjectionMatrix().block(0, 0, dim, dim);
                slice = (int8_t*) _projMatrices.ptr + i * projPitch * dim;
                if (cudaMemcpy2D(slice, projPitch, P.data(), dim * sizeof(real_t),
                                 dim * sizeof(real_t), dim, cudaMemcpyHostToDevice)
                    != cudaSuccess)
                    throw std::logic_error("JosephsMethodCUDA: Could not transfer "
                                           "projection matrices to GPU.");
            }

            int8_t* rayPtr = rayBasePtr + i * rayPitch;
            // get ray origin using direct inverse
            RealVector_t ro =
                -geometryList[i].getInverseProjectionMatrix().block(0, 0, dim, dim)
                * geometryList[i].getProjectionMatrix().block(0, static_cast<index_t>(dim), dim, 1);
            // transfer ray origin
            if (cudaMemcpyAsync(rayPtr, ro.data(), dim * sizeof(real_t), cudaMemcpyHostToDevice)
                != cudaSuccess)
                throw std::logic_error("JosephsMethodCUDA: Could not transfer ray origins to GPU.");
        }
    }

    template <typename data_t>
    JosephsMethodCUDA<data_t>::~JosephsMethodCUDA()
    {
        // Free CUDA resources
        if (cudaFree(_rayOrigins.ptr) != cudaSuccess
            || cudaFree(_projInvMatrices.ptr) != cudaSuccess)
            Logger::get("JosephsMethodCUDA")
                ->error("Couldn't free GPU memory; This may cause problems later.");

        if (_fast)
            if (cudaFree(_projMatrices.ptr) != cudaSuccess)
                Logger::get("JosephsMethodCUDA")
                    ->error("Couldn't free GPU memory; This may cause problems later.");
    }

    template <typename data_t>
    JosephsMethodCUDA<data_t>* JosephsMethodCUDA<data_t>::cloneImpl() const
    {
        return new JosephsMethodCUDA(*this);
    }

    template <typename data_t>
    bool JosephsMethodCUDA<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {
        // TODO: only need Geometry vector stored internally for comparisons, use kernels instead
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
    void JosephsMethodCUDA<data_t>::applyImpl(const DataContainer<data_t>& x,
                                              DataContainer<data_t>& Ax) const
    {
        Timer<> timeGuard("JosephsMethodCUDA", "apply");

        // transfer volume as texture
        auto [volumeTex, volume] = copyTextureToGPU(x);

        // allocate memory for projections
        cudaPitchedPtr dsinoPtr;
        index_t dim = _domainDescriptor->getNumberOfDimensions();
        IndexVector_t rangeDims = _rangeDescriptor->getNumberOfCoefficientsPerDimension();

        auto rangeDimsui = rangeDims.template cast<unsigned int>();

        if (dim == 3) {
            cudaExtent sinoExt =
                make_cudaExtent(rangeDimsui[0] * sizeof(data_t), rangeDimsui[1], rangeDimsui[2]);
            if (cudaMalloc3D(&dsinoPtr, sinoExt) != cudaSuccess)
                throw std::bad_alloc();

            IndexVector_t bmax = _boundingBox._max.template cast<index_t>();
            typename TraverseJosephsCUDA<data_t, 3>::BoundingBox boxMax;
            boxMax._max[0] = static_cast<real_t>(bmax[0]);
            boxMax._max[1] = static_cast<real_t>(bmax[1]);
            boxMax._max[2] = static_cast<real_t>(bmax[2]);

            // synchronize because we are using multiple streams
            cudaDeviceSynchronize();
            // perform projection
            const dim3 sinogramDims(rangeDimsui[2], rangeDimsui[1], rangeDimsui[0]);
            TraverseJosephsCUDA<data_t, 3>::traverseForward(
                sinogramDims, THREADS_PER_BLOCK, volumeTex, (int8_t*) dsinoPtr.ptr, dsinoPtr.pitch,
                (int8_t*) _rayOrigins.ptr, static_cast<uint32_t>(_rayOrigins.pitch),
                (int8_t*) _projInvMatrices.ptr, static_cast<uint32_t>(_projInvMatrices.pitch),
                boxMax);
            cudaDeviceSynchronize();

            // retrieve results from GPU
            copy3DDataContainer<cudaMemcpyDeviceToHost, false>((void*) &Ax[0], dsinoPtr, sinoExt);
        } else {
            if (cudaMallocPitch(&dsinoPtr.ptr, &dsinoPtr.pitch, rangeDimsui[0] * sizeof(data_t),
                                rangeDimsui[1])
                != cudaSuccess)
                throw std::bad_alloc();

            IndexVector_t bmax = _boundingBox._max.template cast<index_t>();
            typename TraverseJosephsCUDA<data_t, 2>::BoundingBox boxMax;
            boxMax._max[0] = static_cast<real_t>(bmax[0]);
            boxMax._max[1] = static_cast<real_t>(bmax[1]);

            // synchronize because we are using multiple streams
            cudaDeviceSynchronize();
            const dim3 sinogramDims(rangeDimsui[1], 1, rangeDimsui[0]);
            TraverseJosephsCUDA<data_t, 2>::traverseForward(
                sinogramDims, THREADS_PER_BLOCK, volumeTex, (int8_t*) dsinoPtr.ptr, dsinoPtr.pitch,
                (int8_t*) _rayOrigins.ptr, static_cast<uint32_t>(_rayOrigins.pitch),
                (int8_t*) _projInvMatrices.ptr, static_cast<uint32_t>(_projInvMatrices.pitch),
                boxMax);
            cudaDeviceSynchronize();

            // retrieve results from GPU
            if (cudaMemcpy2D((void*) &Ax[0], rangeDimsui[0] * sizeof(data_t), dsinoPtr.ptr,
                             dsinoPtr.pitch, rangeDimsui[0] * sizeof(data_t), rangeDimsui[1],
                             cudaMemcpyDeviceToHost)
                != cudaSuccess)
                throw std::logic_error(
                    "JosephsMethodCUDA::apply: Couldn't retrieve results from GPU");
        }

        if (cudaDestroyTextureObject(volumeTex) != cudaSuccess)
            Logger::get("JosephsMethodCUDA")
                ->error("Couldn't destroy texture object; This may cause problems later.");

        if (cudaFreeArray(volume) != cudaSuccess)
            Logger::get("JosephsMethodCUDA")
                ->error("Couldn't free GPU memory; This may cause problems later.");

        if (cudaFree(dsinoPtr.ptr) != cudaSuccess)
            Logger::get("JosephsMethodCUDA")
                ->error("Couldn't free GPU memory; This may cause problems later.");
    }

    template <typename data_t>
    void JosephsMethodCUDA<data_t>::applyAdjointImpl(const DataContainer<data_t>& y,
                                                     DataContainer<data_t>& Aty) const
    {
        Timer<> timeguard("JosephsMethodCUDA", "applyAdjoint");

        // allocate memory for volume
        auto domainDims = _domainDescriptor->getNumberOfCoefficientsPerDimension();
        auto domainDimsui = domainDims.template cast<unsigned int>();

        auto rangeDims = _rangeDescriptor->getNumberOfCoefficientsPerDimension();
        auto rangeDimsui = rangeDims.template cast<unsigned int>();

        index_t dim = _domainDescriptor->getNumberOfDimensions();

        cudaPitchedPtr dvolumePtr;
        if (dim == 3) {
            cudaExtent volExt =
                make_cudaExtent(domainDimsui[0] * sizeof(data_t), domainDimsui[1], domainDimsui[2]);
            if (cudaMalloc3D(&dvolumePtr, volExt) != cudaSuccess)
                throw std::bad_alloc();

            // transfer projections
            if (_fast) {
                auto [sinoTex, sino] = copyTextureToGPU<cudaArrayLayered>(y);

                cudaDeviceSynchronize();
                const dim3 volumeDims(domainDimsui[2], domainDimsui[1], domainDimsui[0]);
                const int threads = THREADS_PER_BLOCK;

                TraverseJosephsCUDA<data_t, 3>::traverseAdjointFast(
                    volumeDims, threads, (int8_t*) dvolumePtr.ptr, dvolumePtr.pitch, sinoTex,
                    (int8_t*) _rayOrigins.ptr, static_cast<uint32_t>(_rayOrigins.pitch),
                    (int8_t*) _projMatrices.ptr, static_cast<uint32_t>(_projMatrices.pitch),
                    rangeDimsui[2]);
                cudaDeviceSynchronize();

                if (cudaDestroyTextureObject(sinoTex) != cudaSuccess)
                    Logger::get("JosephsMethodCUDA")
                        ->error("Couldn't destroy texture object; This may cause problems later.");

                if (cudaFreeArray(sino) != cudaSuccess)
                    Logger::get("JosephsMethodCUDA")
                        ->error("Couldn't free GPU memory; This may cause problems later.");
            } else {
                if (cudaMemset3DAsync(dvolumePtr, 0, volExt) != cudaSuccess)
                    throw std::logic_error("JosephsMethodCUDA::applyAdjoint: Could not "
                                           "zero-initialize volume on GPU.");

                cudaPitchedPtr dsinoPtr;
                cudaExtent sinoExt = make_cudaExtent(rangeDimsui[0] * sizeof(data_t),
                                                     rangeDimsui[1], rangeDimsui[2]);

                if (cudaMalloc3D(&dsinoPtr, sinoExt) != cudaSuccess)
                    throw std::bad_alloc();

                copy3DDataContainer<cudaMemcpyHostToDevice>((void*) &y[0], dsinoPtr, sinoExt);

                // perform projection

                IndexVector_t bmax = _boundingBox._max.template cast<index_t>();
                typename TraverseJosephsCUDA<data_t, 3>::BoundingBox boxMax;
                boxMax._max[0] = static_cast<real_t>(bmax[0]);
                boxMax._max[1] = static_cast<real_t>(bmax[1]);
                boxMax._max[2] = static_cast<real_t>(bmax[2]);

                // synchronize because we are using multiple streams
                cudaDeviceSynchronize();
                const dim3 sinogramDims(rangeDimsui[2], rangeDimsui[1], rangeDimsui[0]);
                TraverseJosephsCUDA<data_t, 3>::traverseAdjoint(
                    sinogramDims, THREADS_PER_BLOCK, (int8_t*) dvolumePtr.ptr, dvolumePtr.pitch,
                    (int8_t*) dsinoPtr.ptr, dsinoPtr.pitch, (int8_t*) _rayOrigins.ptr,
                    static_cast<uint32_t>(_rayOrigins.pitch), (int8_t*) _projInvMatrices.ptr,
                    static_cast<uint32_t>(_projInvMatrices.pitch), boxMax);
                // synchonize because we are using multiple streams
                cudaDeviceSynchronize();

                // free allocated memory
                if (cudaFree(dsinoPtr.ptr) != cudaSuccess)
                    Logger::get("JosephsMethodCUDA")
                        ->error("Couldn't free GPU memory; This may cause problems later.");
            }

            // retrieve results from GPU
            copy3DDataContainer<cudaMemcpyDeviceToHost, false>((void*) &Aty[0], dvolumePtr, volExt);
        } else {
            if (cudaMallocPitch(&dvolumePtr.ptr, &dvolumePtr.pitch,
                                domainDimsui[0] * sizeof(data_t), domainDimsui[1])
                != cudaSuccess)
                throw std::bad_alloc();

            if (_fast) {
                // transfer projections
                auto [sinoTex, sino] = copyTextureToGPU<cudaArrayLayered>(y);

                cudaDeviceSynchronize();
                const dim3 volumeDims(1, domainDimsui[1], domainDimsui[0]);
                const int threads = THREADS_PER_BLOCK;

                TraverseJosephsCUDA<data_t, 2>::traverseAdjointFast(
                    volumeDims, threads, (int8_t*) dvolumePtr.ptr, dvolumePtr.pitch, sinoTex,
                    (int8_t*) _rayOrigins.ptr, static_cast<uint32_t>(_rayOrigins.pitch),
                    (int8_t*) _projMatrices.ptr, static_cast<uint32_t>(_projMatrices.pitch),
                    rangeDimsui[dim - 1]);
                cudaDeviceSynchronize();

                if (cudaDestroyTextureObject(sinoTex) != cudaSuccess)
                    Logger::get("JosephsMethodCUDA")
                        ->error("Couldn't destroy texture object; This may cause problems later.");

                if (cudaFreeArray(sino) != cudaSuccess)
                    Logger::get("JosephsMethodCUDA")
                        ->error("Couldn't free GPU memory; This may cause problems later.");
            } else {
                if (cudaMemset2DAsync(dvolumePtr.ptr, dvolumePtr.pitch, 0,
                                      domainDimsui[0] * sizeof(data_t), domainDimsui[1])
                    != cudaSuccess)
                    throw std::logic_error("JosephsMethodCUDA::applyAdjoint: Could not "
                                           "zero-initialize volume on GPU.");

                cudaPitchedPtr dsinoPtr;
                IndexVector_t rangeDims = _rangeDescriptor->getNumberOfCoefficientsPerDimension();
                if (cudaMallocPitch(&dsinoPtr.ptr, &dsinoPtr.pitch, rangeDimsui[0] * sizeof(data_t),
                                    rangeDimsui[1])
                    != cudaSuccess)
                    throw std::bad_alloc();

                if (cudaMemcpy2DAsync(dsinoPtr.ptr, dsinoPtr.pitch, (void*) &y[0],
                                      rangeDimsui[0] * sizeof(data_t),
                                      rangeDimsui[0] * sizeof(data_t), rangeDimsui[1],
                                      cudaMemcpyHostToDevice)
                    != cudaSuccess)
                    throw std::logic_error(
                        "JosephsMethodCUDA::applyAdjoint: Couldn't transfer sinogram to GPU.");

                IndexVector_t bmax = _boundingBox._max.template cast<index_t>();
                typename TraverseJosephsCUDA<data_t, 2>::BoundingBox boxMax;
                boxMax._max[0] = static_cast<real_t>(bmax[0]);
                boxMax._max[1] = static_cast<real_t>(bmax[1]);
                // synchronize because we are using multiple streams
                cudaDeviceSynchronize();
                const dim3 sinogramDims(rangeDimsui[1], 1, rangeDimsui[0]);
                const int threads = THREADS_PER_BLOCK;
                TraverseJosephsCUDA<data_t, 2>::traverseAdjoint(
                    sinogramDims, threads, (int8_t*) dvolumePtr.ptr, dvolumePtr.pitch,
                    (int8_t*) dsinoPtr.ptr, dsinoPtr.pitch, (int8_t*) _rayOrigins.ptr,
                    static_cast<uint32_t>(_rayOrigins.pitch), (int8_t*) _projInvMatrices.ptr,
                    static_cast<uint32_t>(_projInvMatrices.pitch), boxMax);
                // synchonize because we are using multiple streams
                cudaDeviceSynchronize();

                // free allocated memory
                if (cudaFree(dsinoPtr.ptr) != cudaSuccess)
                    Logger::get("JosephsMethodCUDA")
                        ->error("Couldn't free GPU memory; This may cause problems later.");
            }

            // retrieve results from GPU
            if (cudaMemcpy2D((void*) &Aty[0], domainDimsui[0] * sizeof(data_t), dvolumePtr.ptr,
                             dvolumePtr.pitch, sizeof(data_t) * domainDimsui[0], domainDimsui[1],
                             cudaMemcpyDeviceToHost)
                != cudaSuccess)
                throw std::logic_error(
                    "JosephsMethodCUDA::applyAdjoint: Couldn't retrieve results from GPU");
        }

        // free allocated memory
        if (cudaFree(dvolumePtr.ptr) != cudaSuccess)
            Logger::get("JosephsMethodCUDA")
                ->error("Couldn't free GPU memory; This may cause problems later.");
    }

    template <typename data_t>
    JosephsMethodCUDA<data_t>::JosephsMethodCUDA(const JosephsMethodCUDA<data_t>& other)
        : LinearOperator<data_t>(*other._domainDescriptor, *other._rangeDescriptor),
          _boundingBox{other._boundingBox},
          _geometryList{other._geometryList},
          _fast{other._fast}
    {
        auto dim = static_cast<std::size_t>(_domainDescriptor->getNumberOfDimensions());
        auto numAngles = static_cast<std::size_t>(
            _rangeDescriptor->getNumberOfCoefficientsPerDimension()[static_cast<index_t>(dim) - 1]);

        cudaExtent extent = make_cudaExtent(dim * sizeof(real_t), dim, numAngles);

        if (cudaMallocPitch(&_rayOrigins.ptr, &_rayOrigins.pitch, dim * sizeof(real_t), numAngles)
            != cudaSuccess)
            throw std::bad_alloc();

        _rayOrigins.xsize = dim;
        _rayOrigins.ysize = _geometryList.size();

        if (cudaMalloc3D(&_projInvMatrices, extent) != cudaSuccess)
            throw std::bad_alloc();

        if (cudaMemcpyAsync(_projInvMatrices.ptr, other._projInvMatrices.ptr,
                            _projInvMatrices.pitch * dim * numAngles, cudaMemcpyDeviceToDevice)
            != cudaSuccess)
            throw std::logic_error(
                "JosephsMethodCUDA: Could not transfer inverse projection matrices to GPU.");

        if (cudaMemcpyAsync(_rayOrigins.ptr, other._rayOrigins.ptr, _rayOrigins.pitch * numAngles,
                            cudaMemcpyDeviceToDevice)
            != cudaSuccess)
            throw std::logic_error("JosephsMethodCUDA: Could not transfer ray origins to GPU.");

        if (_fast) {
            if (cudaMalloc3D(&_projMatrices, extent) != cudaSuccess)
                throw std::bad_alloc();

            if (cudaMemcpyAsync(_projMatrices.ptr, other._projMatrices.ptr,
                                _projMatrices.pitch * dim * numAngles, cudaMemcpyDeviceToDevice)
                != cudaSuccess)
                throw std::logic_error(
                    "JosephsMethodCUDA: Could not transfer projection matrices to GPU.");
        }
    }

    template <typename data_t>
    template <cudaMemcpyKind direction, bool async>
    void JosephsMethodCUDA<data_t>::copy3DDataContainer(void* hostData,
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

    template <typename data_t>
    template <unsigned int flags>
    std::pair<cudaTextureObject_t, cudaArray*>
        JosephsMethodCUDA<data_t>::copyTextureToGPU(const DataContainer<data_t>& hostData) const
    {
        // transfer volume as texture
        auto domainDims = hostData.getDataDescriptor().getNumberOfCoefficientsPerDimension();
        auto domainDimsui = domainDims.template cast<unsigned int>();

        cudaArray* volume;
        cudaTextureObject_t volumeTex = 0;

        cudaChannelFormatDesc channelDesc;

        if constexpr (sizeof(data_t) == 4)
            channelDesc =
                cudaCreateChannelDesc(sizeof(data_t) * 8, 0, 0, 0, cudaChannelFormatKindFloat);
        else if (sizeof(data_t) == 8)
            channelDesc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindSigned);
        else
            throw std::invalid_argument("JosephsMethodCUDA::copyTextureToGPU: only supports "
                                        "DataContainer<data_t> with data_t of length 4 or 8 bytes");

        if (hostData.getDataDescriptor().getNumberOfDimensions() == 3) {
            cudaExtent volumeExtent =
                make_cudaExtent(domainDimsui[0], domainDimsui[1], domainDimsui[2]);
            if (cudaMalloc3DArray(&volume, &channelDesc, volumeExtent, flags) != cudaSuccess)
                throw std::bad_alloc();
            cudaMemcpy3DParms cpyParams = {};
            cpyParams.srcPtr.ptr = (void*) &hostData[0];
            cpyParams.srcPtr.pitch = domainDimsui[0] * sizeof(data_t);
            cpyParams.srcPtr.xsize = domainDimsui[0] * sizeof(data_t);
            cpyParams.srcPtr.ysize = domainDimsui[1];
            cpyParams.dstArray = volume;
            cpyParams.extent = volumeExtent;
            cpyParams.kind = cudaMemcpyHostToDevice;

            if (cudaMemcpy3DAsync(&cpyParams) != cudaSuccess)
                throw std::logic_error(
                    "JosephsMethodCUDA::copyTextureToGPU: Could not transfer data to GPU.");
        } else {
            // CUDA has a very weird way of handling layered 1D arrays
            if (flags == cudaArrayLayered) {

                // must be allocated as a 3D Array of height 0
                cudaExtent volumeExtent = make_cudaExtent(domainDimsui[0], 0, domainDimsui[1]);
                if (cudaMalloc3DArray(&volume, &channelDesc, volumeExtent, flags) != cudaSuccess)
                    throw std::bad_alloc();

                // adjust height to 1 for copy
                volumeExtent.height = 1;
                cudaMemcpy3DParms cpyParams = {};
                cpyParams.srcPos = make_cudaPos(0, 0, 0);
                cpyParams.dstPos = make_cudaPos(0, 0, 0);
                cpyParams.srcPtr = make_cudaPitchedPtr(
                    (void*) &hostData[0], domainDimsui[0] * sizeof(data_t), domainDimsui[0], 1);
                cpyParams.dstArray = volume;
                cpyParams.extent = volumeExtent;
                cpyParams.kind = cudaMemcpyHostToDevice;

                if (cudaMemcpy3DAsync(&cpyParams) != cudaSuccess)
                    throw std::logic_error(
                        "JosephsMethodCUDA::copyTextureToGPU: Could not transfer data to GPU.");
            } else {
                if (cudaMallocArray(&volume, &channelDesc, domainDimsui[0], domainDimsui[1], flags)
                    != cudaSuccess)
                    throw std::bad_alloc();

                if (cudaMemcpy2DToArrayAsync(
                        volume, 0, 0, &hostData[0], domainDimsui[0] * sizeof(data_t),
                        domainDimsui[0] * sizeof(data_t), domainDimsui[1], cudaMemcpyHostToDevice)
                    != cudaSuccess)
                    throw std::logic_error(
                        "JosephsMethodCUDA::copyTextureToGPU: Could not transfer data to GPU.");
            }
        }

        cudaResourceDesc resDesc;
        std::memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = volume;

        cudaTextureDesc texDesc;
        std::memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = flags ? cudaAddressModeBorder : cudaAddressModeClamp;
        texDesc.addressMode[1] = flags ? cudaAddressModeBorder : cudaAddressModeClamp;
        texDesc.addressMode[2] = flags ? cudaAddressModeBorder : cudaAddressModeClamp;
        texDesc.filterMode = sizeof(data_t) == 4 ? cudaFilterModeLinear : cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        if (cudaCreateTextureObject(&volumeTex, &resDesc, &texDesc, nullptr) != cudaSuccess)
            throw std::logic_error("Couldn't create texture object");

        return std::pair<cudaTextureObject_t, cudaArray*>(volumeTex, volume);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class JosephsMethodCUDA<float>;
    template class JosephsMethodCUDA<double>;
} // namespace elsa