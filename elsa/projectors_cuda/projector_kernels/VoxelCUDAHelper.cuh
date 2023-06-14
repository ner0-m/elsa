#pragma once

#include "elsaDefines.h"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/universal_vector.h>

#include "DataContainer.h"
#include "ProjectVoxelsCUDA.cuh"
#include "DetectorDescriptor.h"
#include "Geometry.h"
#include "Luts.hpp"

namespace elsa
{

    template <typename data_t>
    void transferGeometries(const std::vector<Geometry>& geometries,
                            thrust::device_vector<data_t>& _projMatrices,
                            thrust::device_vector<data_t>& _extMatrices)
    {
        if (geometries.empty()) {
            throw std::runtime_error("No geometries provided.");
        }
        // allocate device memory and copy extrinsic and projection matrices to device
        // height = dim + 1 because we are using homogenous coordinates
        index_t dim = geometries[0].getDimension();
        size_t width = dim, height = dim + 1, depth = geometries.size();

        // allocate device memory and copy ray origins and the inverse of the significant part
        // of projection matrices to device
        _projMatrices.resize(width * height * depth);
        _extMatrices.resize(width * height * depth);
        auto projMatIter = _projMatrices.begin();
        auto extMatIter = _extMatrices.begin();

        for (const auto& geometry : geometries) {

            RealMatrix_t P = geometry.getProjectionMatrix();
            RealMatrix_t E = geometry.getExtrinsicMatrix();

            // CUDA also uses a column-major representation, directly transfer matrix
            // transfer projection and extrinsic matrix
            projMatIter = thrust::copy(P.data(), P.data() + P.size(), projMatIter);
            extMatIter = thrust::copy(E.data(), E.data() + E.size(), extMatIter);
        }
    }

    template <typename data_t, size_t dim, bool adjoint,
              VoxelHelperCUDA::PROJECTOR_TYPE type = VoxelHelperCUDA::CLASSIC,
              size_t N = DEFAULT_PROJECTOR_LUT_SIZE>
    void projectVoxelsCUDA(const DataContainer<data_t>& volumeContainer,
                           const DataContainer<data_t>& sinoContainer, const Lut<data_t, N>& lut,
                           const thrust::device_vector<data_t>& _projMatrices,
                           const thrust::device_vector<data_t>& _extMatrices)
    {
        auto domainDims = volumeContainer.getDataDescriptor().getNumberOfCoefficientsPerDimension();
        auto domainDimsui = domainDims.template cast<unsigned int>();
        IndexVector_t rangeDims =
            sinoContainer.getDataDescriptor().getNumberOfCoefficientsPerDimension();
        auto rangeDimsui = rangeDims.template cast<unsigned int>();
        auto& volume = volumeContainer.storage();
        auto& sino = sinoContainer.storage();
        auto* dvolume = thrust::raw_pointer_cast(volume.data());
        auto* dsino = thrust::raw_pointer_cast(sino.data());
        const auto* projMat = thrust::raw_pointer_cast(_projMatrices.data());
        const auto* extMat = thrust::raw_pointer_cast(_extMatrices.data());

        // copy lut to device
        thrust::device_vector<data_t> lutArray(lut.size());
        auto lutData = lut.data();
        thrust::copy(lutData.begin(), lutData.end(), lutArray.begin());
        auto* dlut = thrust::raw_pointer_cast(lutArray.data());

        // assume that all geometries are from the same setup
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
        cudaMemAdvise(dlut, lutArray.size(), cudaMemAdviseSetReadMostly, device);

        if constexpr (adjoint) {
            // Advice, that sinogram is read only
            cudaMemAdvise(dsino, sino.size(), cudaMemAdviseSetReadMostly, device);
            // Advice, that volume is not read only
            cudaMemAdvise(dvolume, volume.size(), cudaMemAdviseUnsetReadMostly, device);
        } else {
            // Advice, that sinogram is not read only
            cudaMemAdvise(dsino, sino.size(), cudaMemAdviseUnsetReadMostly, device);
            // Advice, that volume is read only
            cudaMemAdvise(dvolume, volume.size(), cudaMemAdviseSetReadMostly, device);
        }

        if (dim == 3) {
            dim3 sinogramDims(rangeDimsui[0], rangeDimsui[1], rangeDimsui[2]);
            dim3 volumeDims(domainDimsui[0], domainDimsui[1], domainDimsui[2]);
            ProjectVoxelsCUDA<data_t, 3, adjoint, type>::project(
                volumeDims, sinogramDims, ProjectVoxelsCUDA<data_t>::MAX_THREADS_PER_BLOCK,
                const_cast<data_t*>(dvolume), const_cast<data_t*>(dsino), projMat, extMat,
                const_cast<data_t*>(dlut), lut.support(), sourceDetectorDistance);
        } else {
            dim3 sinogramDims(rangeDimsui[0], 1, rangeDimsui[1]);
            dim3 volumeDims(domainDimsui[0], domainDimsui[1], 1);
            ProjectVoxelsCUDA<data_t, 2, adjoint, type>::project(
                volumeDims, sinogramDims, ProjectVoxelsCUDA<data_t>::MAX_THREADS_PER_BLOCK,
                const_cast<data_t*>(dvolume), const_cast<data_t*>(dsino), projMat, extMat,
                const_cast<data_t*>(dlut), lut.support(), sourceDetectorDistance);
        }
        // synchonize because we are using multiple streams
        cudaDeviceSynchronize();
    }
} // namespace elsa