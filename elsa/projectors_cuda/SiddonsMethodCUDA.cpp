#include "SiddonsMethodCUDA.h"
#include "LogGuard.h"
#include "Timer.h"
#include "TypeCasts.hpp"
#include "Logger.h"

#include <thrust/copy.h>
#include <thrust/host_vector.h>

namespace elsa
{
    template <typename data_t>
    SiddonsMethodCUDA<data_t>::SiddonsMethodCUDA(const VolumeDescriptor& domainDescriptor,
                                                 const DetectorDescriptor& rangeDescriptor)
        : base_type(domainDescriptor, rangeDescriptor)
    {
        const auto& detectorDescriptor = static_cast<const DetectorDescriptor&>(*_rangeDescriptor);
        auto dim = static_cast<std::size_t>(_domainDescriptor->getNumberOfDimensions());
        if (dim != static_cast<std::size_t>(_rangeDescriptor->getNumberOfDimensions())) {
            throw LogicError(
                std::string("SiddonsMethodCUDA: domain and range dimension need to match"));
        }

        if (dim != 2 && dim != 3) {
            throw LogicError("SiddonsMethodCUDA: only supporting 2d/3d operations");
        }

        if (detectorDescriptor.getNumberOfGeometryPoses() == 0) {
            throw LogicError("SiddonsMethodCUDA: geometry list was empty");
        }

        const auto numGeometry = asUnsigned(detectorDescriptor.getNumberOfGeometryPoses());

        // allocate device memory and copy ray origins and the inverse of the significant part of
        // projection matrices to device
        _invProjMatrices.resize(dim * dim * numGeometry);
        auto matricesIter = _invProjMatrices.begin();

        // Allocate memory for ray origins
        _rayOrigins.resize(dim * numGeometry);
        auto originsIter = _rayOrigins.begin();

        const auto& poses = detectorDescriptor.getGeometry();
        for (std::size_t i = 0; i < poses.size(); ++i) {
            const auto& geometry = poses[i];

            RealMatrix_t P = geometry.getInverseProjectionMatrix().block(0, 0, dim, dim);

            // CUDA also uses a column-major representation, directly transfer matrix
            // transfer inverse of projection matrix
            matricesIter = thrust::copy(P.data(), P.data() + P.size(), matricesIter);

            // get camera center using direct inverse
            RealVector_t ro = geometry.getCameraCenter();

            // transfer ray origin
            originsIter = thrust::copy(ro.begin(), ro.end(), originsIter);
        }
    }

    template <typename data_t>
    SiddonsMethodCUDA<data_t>* SiddonsMethodCUDA<data_t>::_cloneImpl() const
    {
        return new SiddonsMethodCUDA<data_t>(static_cast<VolumeDescriptor&>(*_domainDescriptor),
                                             static_cast<DetectorDescriptor&>(*_rangeDescriptor));
    }

    template <typename data_t>
    void SiddonsMethodCUDA<data_t>::forward(const BoundingBox& aabb, const DataContainer<data_t>& x,
                                            DataContainer<data_t>& Ax) const
    {
        Timer timeGuard("SiddonsMethodCUDA", "apply");

        traverseVolume<false>(aabb, (void*) &(x[0]), (void*) &(Ax[0]));
    }

    template <typename data_t>
    void SiddonsMethodCUDA<data_t>::backward(const BoundingBox& aabb,
                                             const DataContainer<data_t>& y,
                                             DataContainer<data_t>& Aty) const
    {
        Timer timeguard("SiddonsMethodCUDA", "applyAdjoint");

        traverseVolume<true>(aabb, (void*) &(Aty[0]), (void*) &(y[0]));
    }

    template <typename data_t>
    bool SiddonsMethodCUDA<data_t>::_isEqual(const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        auto otherSM = downcast_safe<SiddonsMethodCUDA>(&other);
        return static_cast<bool>(otherSM);
    }

    template <typename data_t>
    template <bool adjoint>
    void SiddonsMethodCUDA<data_t>::traverseVolume(const BoundingBox& aabb, void* volumePtr,
                                                   void* sinoPtr) const
    {
        auto domainDims = _domainDescriptor->getNumberOfCoefficientsPerDimension();
        auto domainDimsui = domainDims.template cast<unsigned int>();
        IndexVector_t rangeDims = _rangeDescriptor->getNumberOfCoefficientsPerDimension();
        auto rangeDimsui = rangeDims.template cast<unsigned int>();
        cudaPitchedPtr dvolumePtr;
        cudaPitchedPtr dsinoPtr;

        if (_domainDescriptor->getNumberOfDimensions() == 3) {
            typename TraverseSiddonsCUDA<data_t, 3>::BoundingBox boundingBox;
            boundingBox._max[0] = aabb.max().template cast<uint32_t>()[0];
            boundingBox._max[1] = aabb.max().template cast<uint32_t>()[1];
            boundingBox._max[2] = aabb.max().template cast<uint32_t>()[2];

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
                copy3DDataContainerGPU<ContainerCpyKind::cpyContainerToRawGPU>(sinoPtr, dsinoPtr,
                                                                               sinoExt);
                if (cudaMemset3DAsync(dvolumePtr, 0, volExt) != cudaSuccess)
                    throw LogicError("SiddonsMethodCUDA::traverseVolume: Could not "
                                     "zero-initialize volume on GPU.");
            } else {
                copy3DDataContainerGPU<ContainerCpyKind::cpyContainerToRawGPU>(volumePtr,
                                                                               dvolumePtr, volExt);
            }

            dim3 sinogramDims(rangeDimsui[2], rangeDimsui[1], rangeDimsui[0]);
            // synchronize because we are using multiple streams
            cudaDeviceSynchronize();
            if (adjoint) {
                TraverseSiddonsCUDA<data_t, 3>::traverseAdjoint(
                    sinogramDims, THREADS_PER_BLOCK, (int8_t*) dvolumePtr.ptr, dvolumePtr.pitch,
                    (int8_t*) dsinoPtr.ptr, dsinoPtr.pitch,
                    thrust::raw_pointer_cast(_rayOrigins.data()),
                    thrust::raw_pointer_cast(_invProjMatrices.data()), boundingBox);
            } else {
                TraverseSiddonsCUDA<data_t, 3>::traverseForward(
                    sinogramDims, THREADS_PER_BLOCK, (int8_t*) dvolumePtr.ptr, dvolumePtr.pitch,
                    (int8_t*) dsinoPtr.ptr, dsinoPtr.pitch,
                    thrust::raw_pointer_cast(_rayOrigins.data()),
                    thrust::raw_pointer_cast(_invProjMatrices.data()), boundingBox);
            }
            // synchonize because we are using multiple streams
            cudaDeviceSynchronize();

            // retrieve results from GPU
            if (adjoint)
                copy3DDataContainerGPU<ContainerCpyKind::cpyRawGPUToContainer, false>(
                    volumePtr, dvolumePtr, volExt);
            else
                copy3DDataContainerGPU<ContainerCpyKind::cpyRawGPUToContainer, false>(
                    sinoPtr, dsinoPtr, sinoExt);

        } else {
            typename TraverseSiddonsCUDA<data_t, 2>::BoundingBox boundingBox;
            boundingBox._max[0] = aabb.max().template cast<uint32_t>()[0];
            boundingBox._max[1] = aabb.max().template cast<uint32_t>()[1];

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
                        rangeDimsui[0] * sizeof(data_t), rangeDimsui[1], cudaMemcpyDefault)
                    != cudaSuccess)
                    throw LogicError(
                        "SiddonsMethodCUDA::traverseVolume: Couldn't transfer sinogram to GPU.");

                if (cudaMemset2DAsync(dvolumePtr.ptr, dvolumePtr.pitch, 0,
                                      domainDimsui[0] * sizeof(data_t), domainDimsui[1])
                    != cudaSuccess)
                    throw LogicError("SiddonsMethodCUDA::traverseVolume: Could not "
                                     "zero-initialize volume on GPU.");
            } else {
                if (cudaMemcpy2DAsync(dvolumePtr.ptr, dvolumePtr.pitch, volumePtr,
                                      domainDimsui[0] * sizeof(data_t),
                                      domainDimsui[0] * sizeof(data_t), domainDimsui[1],
                                      cudaMemcpyDefault)
                    != cudaSuccess)
                    throw LogicError(
                        "SiddonsMethodCUDA::traverseVolume: Couldn't transfer volume to GPU.");
            }

            // perform projection
            dim3 sinogramDims(rangeDimsui[1], 1, rangeDimsui[0]);
            // synchronize because we are using multiple streams
            cudaDeviceSynchronize();
            if (adjoint) {
                TraverseSiddonsCUDA<data_t, 2>::traverseAdjoint(
                    sinogramDims, THREADS_PER_BLOCK, (int8_t*) dvolumePtr.ptr, dvolumePtr.pitch,
                    (int8_t*) dsinoPtr.ptr, dsinoPtr.pitch,
                    thrust::raw_pointer_cast(_rayOrigins.data()),
                    thrust::raw_pointer_cast(_invProjMatrices.data()), boundingBox);
            } else {
                TraverseSiddonsCUDA<data_t, 2>::traverseForward(
                    sinogramDims, THREADS_PER_BLOCK, (int8_t*) dvolumePtr.ptr, dvolumePtr.pitch,
                    (int8_t*) dsinoPtr.ptr, dsinoPtr.pitch,
                    thrust::raw_pointer_cast(_rayOrigins.data()),
                    thrust::raw_pointer_cast(_invProjMatrices.data()), boundingBox);
            }
            // synchronize because we are using multiple streams
            cudaDeviceSynchronize();

            // retrieve results from GPU
            if (adjoint) {
                if (cudaMemcpy2D(volumePtr, domainDimsui[0] * sizeof(data_t), dvolumePtr.ptr,
                                 dvolumePtr.pitch, domainDimsui[0] * sizeof(data_t),
                                 domainDimsui[1], cudaMemcpyDefault)
                    != cudaSuccess)
                    throw LogicError(
                        "SiddonsMethodCUDA::traverseVolume: Couldn't retrieve results from GPU.");
            } else {
                if (cudaMemcpy2D(sinoPtr, rangeDimsui[0] * sizeof(data_t), dsinoPtr.ptr,
                                 dsinoPtr.pitch, rangeDimsui[0] * sizeof(data_t), rangeDimsui[1],
                                 cudaMemcpyDefault)
                    != cudaSuccess)
                    throw LogicError(
                        "SiddonsMethodCUDA::traverseVolume: Couldn't retrieve results from GPU");
            }
        }

        // free allocated memory
        if (cudaFree(dvolumePtr.ptr) != cudaSuccess || cudaFree(dsinoPtr.ptr) != cudaSuccess)
            Logger::get("SiddonsMethodCUDA")
                ->error("Couldn't free GPU memory; This may cause problems later.");
    }

    template <typename data_t>
    template <typename SiddonsMethodCUDA<data_t>::ContainerCpyKind direction, bool async>
    void SiddonsMethodCUDA<data_t>::copy3DDataContainerGPU(void* hostData,
                                                           const cudaPitchedPtr& gpuData,
                                                           const cudaExtent& extent) const
    {
        cudaMemcpy3DParms cpyParams = {};
        cpyParams.extent = extent;
        cpyParams.kind = cudaMemcpyDefault;

        cudaPitchedPtr tmp =
            make_cudaPitchedPtr(hostData, extent.width, extent.width, extent.height);

        if (direction == ContainerCpyKind::cpyContainerToRawGPU) {
            cpyParams.dstPtr = gpuData;
            cpyParams.srcPtr = tmp;
        } else if (direction == ContainerCpyKind::cpyRawGPUToContainer) {
            cpyParams.srcPtr = gpuData;
            cpyParams.dstPtr = tmp;
        }

        if (async) {
            if (cudaMemcpy3DAsync(&cpyParams) != cudaSuccess)
                throw LogicError("Failed copying data between device and host");
        } else {
            if (cudaMemcpy3D(&cpyParams) != cudaSuccess)
                throw LogicError("Failed copying data between device acudaMemcpyKindnd host");
        }
    }

    // ------------------------------------------
    // explicit template instantiation
    template class SiddonsMethodCUDA<float>;
    template class SiddonsMethodCUDA<double>;
} // namespace elsa
