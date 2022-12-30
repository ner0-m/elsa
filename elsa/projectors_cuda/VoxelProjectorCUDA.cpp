#include "SiddonsMethodCUDA.h"
#include "LogGuard.h"
#include "Timer.h"
#include "TypeCasts.hpp"

#include "Logger.h"
#include "VoxelProjectorCUDA.h"

#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace elsa
{
    template <typename data_t>
    VoxelProjectorCUDA<data_t>::VoxelProjectorCUDA(const VolumeDescriptor& domainDescriptor,
                                                   const DetectorDescriptor& rangeDescriptor)
        : VoxelProjectorCUDA(2.f, 10.83f, 2.f, domainDescriptor, rangeDescriptor)
    {
    }

    template <typename data_t>
    VoxelProjectorCUDA<data_t>::VoxelProjectorCUDA(data_t radius, data_t alpha, data_t order,
                                                   const VolumeDescriptor& domainDescriptor,
                                                   const DetectorDescriptor& rangeDescriptor)
        : base_type(domainDescriptor, rangeDescriptor),
          _detectorDescriptor(static_cast<DetectorDescriptor&>(*_rangeDescriptor)),
          _volumeDescriptor(static_cast<VolumeDescriptor&>(*_domainDescriptor)),
          _lut(radius, alpha, order)
    {
        auto dim = static_cast<std::size_t>(_domainDescriptor->getNumberOfDimensions());
        if (dim != static_cast<std::size_t>(_rangeDescriptor->getNumberOfDimensions())) {
            throw LogicError(
                std::string("VoxelProjectorCUDA: domain and range dimension need to match"));
        }

        if (dim != 2 && dim != 3) {
            throw LogicError("VoxelProjectorCUDA: only supporting 2d/3d operations");
        }

        if (_detectorDescriptor.getNumberOfGeometryPoses() == 0) {
            throw LogicError("VoxelProjectorCUDA: geometry list was empty");
        }

        // allocate device memory and copy extrinsic and projection matrices to device
        // width, height, depth
        size_t width = dim, height = dim + 1,
               depth = asUnsigned(_detectorDescriptor.getNumberOfGeometryPoses());
        cudaExtent extent = make_cudaExtent(width * sizeof(real_t), height, depth);

        if (cudaMalloc3D(&_projMatrices, extent) != cudaSuccess)
            throw std::bad_alloc();
        if (cudaMalloc3D(&_extMatrices, extent) != cudaSuccess)
            throw std::bad_alloc();

        auto* projPtr = (int8_t*) _projMatrices.ptr;
        auto projPitch = _projMatrices.pitch;
        auto* extPtr = (int8_t*) _extMatrices.ptr;
        auto extPitch = _extMatrices.pitch;

        const auto& poses = _detectorDescriptor.getGeometry();
        for (std::size_t i = 0; i < poses.size(); ++i) {
            const auto& geometry = poses[i];

            auto& P = geometry.getProjectionMatrix();
            auto& E = geometry.getExtrinsicMatrix();

            int8_t* projSlice = projPtr + i * projPitch * height;
            int8_t* extSlice = extPtr + i * extPitch * height;

            // CUDA also uses a column-major representation, directly transfer matrix
            // transfer inverse of projection matrix
            if (cudaMemcpy2DAsync(projSlice, projPitch, P.data(), width * sizeof(real_t),
                                  width * sizeof(real_t), height, cudaMemcpyHostToDevice)
                != cudaSuccess)
                throw LogicError(
                    "VoxelProjectorCUDA: Could not transfer projection matrix to GPU.");
            if (cudaMemcpy2DAsync(extSlice, extPitch, E.data(), width * sizeof(real_t),
                                  width * sizeof(real_t), height, cudaMemcpyHostToDevice)
                != cudaSuccess)
                throw LogicError(
                    "VoxelProjectorCUDA: Could not transfer extrinsic projection matrix to GPU.");
        }

        // copy lut to device
        auto lutData = _lut.data();
        _lutArray.resize(sizeof(lutData));
        thrust::copy(lutData.begin(), lutData.end(), _lutArray.begin());
    }

    template <typename data_t>
    VoxelProjectorCUDA<data_t>::~VoxelProjectorCUDA()
    {
        // Free CUDA resources
        if (cudaFree(_projMatrices.ptr) != cudaSuccess || cudaFree(_extMatrices.ptr) != cudaSuccess)
            Logger::get("VoxelProjectorCUDA")
                ->error("Couldn't free GPU memory; This may cause problems later.");
    }

    template <typename data_t>
    VoxelProjectorCUDA<data_t>* VoxelProjectorCUDA<data_t>::_cloneImpl() const
    {
        return new VoxelProjectorCUDA<data_t>(_volumeDescriptor, _detectorDescriptor);
    }

    template <typename data_t>
    void VoxelProjectorCUDA<data_t>::forward(const BoundingBox& aabb,
                                             const DataContainer<data_t>& x,
                                             DataContainer<data_t>& Ax) const
    {
        (void) aabb;
        Timer timeGuard("ProjectVoxelsCUDA", "apply");

        // Set it to zero, just to be sure
        Ax = 0;

        projectForward(x, Ax);
    }

    template <typename data_t>
    void VoxelProjectorCUDA<data_t>::backward(const BoundingBox& aabb,
                                              const DataContainer<data_t>& y,
                                              DataContainer<data_t>& Aty) const
    {
        (void) aabb;
        Timer timeguard("ProjectVoxelsCUDA", "applyAdjoint");

        // Set it to zero, just to be sure
        Aty = 0;

        projectBackward(Aty, y);
    }

    template <typename data_t>
    bool VoxelProjectorCUDA<data_t>::_isEqual(const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        auto otherSM = downcast_safe<VoxelProjectorCUDA>(&other);
        return static_cast<bool>(otherSM);
    }

    template <typename data_t>
    void VoxelProjectorCUDA<data_t>::projectForward(const DataContainer<data_t>& volumeContainer,
                                                    DataContainer<data_t>& sinoContainer) const
    {
        auto domainDims = _domainDescriptor->getNumberOfCoefficientsPerDimension();
        auto domainDimsui = domainDims.template cast<unsigned int>();
        IndexVector_t rangeDims = _rangeDescriptor->getNumberOfCoefficientsPerDimension();
        auto rangeDimsui = rangeDims.template cast<unsigned int>();
        auto& volume = volumeContainer.storage();
        auto& sino = sinoContainer.storage();
        auto* dvolume = thrust::raw_pointer_cast(volume.data());
        auto* dsino = thrust::raw_pointer_cast(sino.data());
        auto* dlut = thrust::raw_pointer_cast(_lutArray.data());

        const DetectorDescriptor& detectorDesc =
            downcast<DetectorDescriptor>(sinoContainer.getDataDescriptor());
        const Geometry& geometry = detectorDesc.getGeometry()[0];
        real_t sourceDetectorDistance = geometry.getSourceDetectorDistance();

        // Prefetch unified memory
        int device = -1;
        cudaGetDevice(&device);
        cudaMemPrefetchAsync(dvolume, volume.size() * sizeof(data_t), device);
        cudaMemPrefetchAsync(dsino, sino.size() * sizeof(data_t), device);

        // advice lut is readonly
        cudaMemAdvise(dlut, _lutArray.size(), cudaMemAdviseSetReadMostly, device);
        // Advice, that sinogram is read only
        cudaMemAdvise(dsino, sino.size(), cudaMemAdviseUnsetReadMostly, device);
        // Advice, that volume is not read only
        cudaMemAdvise(dvolume, volume.size(), cudaMemAdviseSetReadMostly, device);

        if (_domainDescriptor->getNumberOfDimensions() == 3) {
            dim3 sinogramDims(rangeDimsui[0], rangeDimsui[1], rangeDimsui[2]);
            dim3 volumeDims(domainDimsui[0], domainDimsui[1], domainDimsui[2]);
            ProjectVoxelsCUDA<data_t, 3>::forward(
                volumeDims, sinogramDims, THREADS_PER_DIM, const_cast<data_t*>(dvolume), dsino,
                (int8_t*) _projMatrices.ptr, static_cast<uint32_t>(_projMatrices.pitch),
                (int8_t*) _extMatrices.ptr, static_cast<uint32_t>(_extMatrices.pitch),
                const_cast<data_t*>(dlut), _lut.radius(), sourceDetectorDistance);
        } else {
            dim3 sinogramDims(rangeDimsui[0], 1, rangeDimsui[1]);
            dim3 volumeDims(domainDimsui[0], domainDimsui[1], 1);
            ProjectVoxelsCUDA<data_t, 2>::forward(
                volumeDims, sinogramDims, THREADS_PER_DIM, const_cast<data_t*>(dvolume), dsino,
                (int8_t*) _projMatrices.ptr, static_cast<uint32_t>(_projMatrices.pitch),
                (int8_t*) _extMatrices.ptr, static_cast<uint32_t>(_extMatrices.pitch),
                const_cast<data_t*>(dlut), _lut.radius(), sourceDetectorDistance);
        }
        // synchonize because we are using multiple streams
        cudaDeviceSynchronize();
    }

    template <typename data_t>
    void VoxelProjectorCUDA<data_t>::projectBackward(
        DataContainer<data_t>& volumeContainer, const DataContainer<data_t>& sinoContainer) const
    {
        auto domainDims = _domainDescriptor->getNumberOfCoefficientsPerDimension();
        auto domainDimsui = domainDims.template cast<unsigned int>();
        IndexVector_t rangeDims = _rangeDescriptor->getNumberOfCoefficientsPerDimension();
        auto rangeDimsui = rangeDims.template cast<unsigned int>();
        auto& volume = volumeContainer.storage();
        auto& sino = sinoContainer.storage();
        auto* dvolume = thrust::raw_pointer_cast(volume.data());
        auto* dsino = thrust::raw_pointer_cast(sino.data());
        auto* dlut = thrust::raw_pointer_cast(_lutArray.data());

        const DetectorDescriptor& detectorDesc =
            downcast<DetectorDescriptor>(sinoContainer.getDataDescriptor());
        const Geometry& geometry = detectorDesc.getGeometry()[0];
        real_t sourceDetectorDistance = geometry.getSourceDetectorDistance();

        // Prefetch unified memory
        int device = -1;
        cudaGetDevice(&device);
        cudaMemPrefetchAsync(dvolume, volume.size() * sizeof(data_t), device);
        cudaMemPrefetchAsync(dsino, sino.size() * sizeof(data_t), device);

        // advice lut is readonly
        cudaMemAdvise(dlut, _lutArray.size(), cudaMemAdviseSetReadMostly, device);
        // Advice, that sinogram is read only
        cudaMemAdvise(dsino, sino.size(), cudaMemAdviseSetReadMostly, device);
        // Advice, that volume is not read only
        cudaMemAdvise(dvolume, volume.size(), cudaMemAdviseUnsetReadMostly, device);

        if (_domainDescriptor->getNumberOfDimensions() == 3) {
            dim3 sinogramDims(rangeDimsui[0], rangeDimsui[1], rangeDimsui[2]);
            dim3 volumeDims(domainDimsui[0], domainDimsui[1], domainDimsui[2]);
            ProjectVoxelsCUDA<data_t, 3>::backward(
                volumeDims, sinogramDims, THREADS_PER_DIM, dvolume, const_cast<data_t*>(dsino),
                (int8_t*) _projMatrices.ptr, static_cast<uint32_t>(_projMatrices.pitch),
                (int8_t*) _extMatrices.ptr, static_cast<uint32_t>(_extMatrices.pitch),
                const_cast<data_t*>(dlut), _lut.radius(), sourceDetectorDistance);
        } else {
            dim3 sinogramDims(rangeDimsui[0], 1, rangeDimsui[1]);
            dim3 volumeDims(domainDimsui[0], domainDimsui[1], 1);
            ProjectVoxelsCUDA<data_t, 2>::backward(
                volumeDims, sinogramDims, THREADS_PER_DIM, dvolume, const_cast<data_t*>(dsino),
                (int8_t*) _projMatrices.ptr, static_cast<uint32_t>(_projMatrices.pitch),
                (int8_t*) _extMatrices.ptr, static_cast<uint32_t>(_extMatrices.pitch),
                const_cast<data_t*>(dlut), _lut.radius(), sourceDetectorDistance);
        }
        // synchonize because we are using multiple streams
        cudaDeviceSynchronize();
    }

    // ------------------------------------------
    // explicit template instantiation
    template class VoxelProjectorCUDA<float>;
    template class VoxelProjectorCUDA<double>;
} // namespace elsa
