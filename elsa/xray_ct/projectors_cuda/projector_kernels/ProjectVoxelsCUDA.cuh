/**
 * @file VoxelProjectorCUDA.cuh
 *
 * @brief Provides interface definitions for the Voxel CUDA projector. Allows for separable
 * compilation of device and host code.
 *
 */
#pragma once

#include "cuda_runtime.h"
#include <cstdint>
#include "elsaDefines.h"

namespace elsa
{
    namespace VoxelHelperCUDA
    {
        enum PROJECTOR_TYPE { CLASSIC, DIFFERENTIAL };
    }

    template <typename data_t = real_t, uint32_t dim = 3, bool adjoint = false,
              VoxelHelperCUDA::PROJECTOR_TYPE type = VoxelHelperCUDA::CLASSIC>
    struct ProjectVoxelsCUDA {
        const static uint32_t MAX_THREADS_PER_BLOCK = 1024;

        /**
         * @brief Forward projection using voxels
         *
         * @param[in] volumeDims specifies the dimensions of the volume
         * @param[in] numGeoms specifies the number of dimensions
         * @param[in] threads specifies the number of threads for each block
         * @param[in] volume pointer to volume data
         * @param[in] volumePitch pitch (padded width in bytes) of volume
         * @param[out] sinogram pointer to output
         * @param[in] sinogramPitch pitch of sinogram
         * @param[in] proj pointer to projection matrices
         * @param[in] projPitch pitch of projection matrices
         * @param[in] ext pointer to ext matrices
         * @param[in] extPitch pitch of ext matrices
         * @param[in] lut pointer to lookup table
         * @param[in] radius of blob
         * @param[in] sdd source detector distance
         *
         * volumeDims should be given as (volumeSizeX, volumeSizeY, volumeSizeZ).
         *
         * threads should ideally be a multiple of the warp size (32 for all current GPUs).
         */

        static void project(const dim3 volumeDims, const dim3 sinogramDims, const int threads,
                            data_t* __restrict__ volume, data_t* __restrict__ sinogram,
                            const data_t* __restrict__ proj, const data_t* __restrict__ ext,
                            const data_t* __restrict__ lut, const data_t radius, const data_t sdd);
    };
} // namespace elsa