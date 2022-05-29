/**
 * @file TraverseJosephsCUDA.cuh
 *
 * @brief Provides interface definitions for the Joseph's CUDA projector. Allows for separable
 * compilation of device and host code.
 *
 * @author Nikola Dinev (nikola.dinev@tum.de)
 */
#pragma once

#include <cstdint>

#include "elsaDefinesCUDA.cuh"

namespace elsa
{

    template <typename data_t = real_t, uint dim = 3>
    class TraverseJosephsCUDA
    {
    public:
        constexpr static uint32_t MAX_THREADS_PER_BLOCK = 32;

        CUDA_HOST TraverseJosephsCUDA(const BoundingBoxCUDA<dim>& volumeBoundingBox,
                                      const BoundingBoxCUDA<dim>& sinogramBoundingBox,
                                      cudaPitchedPtr rayOrigins, cudaPitchedPtr projInvMatrices,
                                      cudaPitchedPtr projMatrices = {0, 0, 0, 0})
            : _volumeBoundingBox{volumeBoundingBox},
              _sinogramBoundingBox{sinogramBoundingBox},
              _projInvMatrices{projInvMatrices},
              _projMatrices{projMatrices},
              _rayOrigins{rayOrigins}
        {
        }

        CUDA_HOST ~TraverseJosephsCUDA()
        {
            // Free CUDA resources
            cudaFree(_rayOrigins.ptr);
            cudaFree(_projInvMatrices.ptr);
            cudaFree(_projMatrices.ptr);
        }

        /**
         * @brief Forward projection using Josephs's method
         *
         * @param[in] sinogramDims specifies the dimensions of the sinogram
         * @param[in] threads specifies the number of threads for each block
         * @param[in] volume handle to the texture object containing the volume data
         * @param[out] sinogram pointer to output
         * @param[in] sinogramPitch pitch (padded width in bytes) of sinogram
         * @param[in] rayOrigins pointer to ray origins
         * @param[in] originPitch pitch of ray origins
         * @param[in] projInv pointer to inverse of projection matrices (stored column-wise)
         * @param[in] projPitch pitch of inverse of projection matrices
         * @param[in] boundingBox specifies the size of the volume
         *
         * sinogramDims should be given as (numAngles, detectorSizeY (1 for 2D), detectorSizeX).
         *
         * threads should ideally be a multiple of the warp size (32 for all current GPUs).
         *
         * @warning Interpolation mode for texture should be set to cudaFilterModePoint when working
         * with doubles.
         */
        CUDA_HOST void traverseForward(cudaTextureObject_t volume, cudaPitchedPtr& sinogram);

        CUDA_HOST void traverseForwardConstrained(cudaTextureObject_t volume,
                                                  cudaPitchedPtr& sinogram,
                                                  const BoundingBoxCUDA<dim>& volumeBoundingBox,
                                                  const BoundingBoxCUDA<dim - 1>& imageBoundingBox,
                                                  const index_t numPoses,
                                                  const std::vector<Interval>& poses,
                                                  bool zeroInit = true, index_t paddedDim = -1,
                                                  const cudaStream_t& stream = (cudaStream_t) 0);
        /**
         * @brief Backward projection using Josephs's method
         *
         * @param[in] sinogramDims specifies the dimensions of the sinogram
         * @param[in] threads specifies the number of threads for each block
         * @param[out] volume pointer to output
         * @param[in] volumePitch pitch (padded width in bytes) of volume
         * @param[in] sinogram pointer to sinogram data
         * @param[in] sinogramPitch pitch of sinogram
         * @param[in] rayOrigins pointer to ray origins
         * @param[in] originPitch pitch of ray origins
         * @param[in] projInv pointer to inverse of projection matrices
         * @param[in] projPitch pitch of inverse of projection matrices
         * @param[in] boundingBox specifies the size of the volume
         *
         * sinogramDims should be given as (numAngles, detectorSizeY (1 for 2D), detectorSizeX).
         *
         * threads should ideally be a multiple of the warp size (32 for all current GPUs).
         *
         * This is the matched version of the forward traversal. Considerably slower
         * as interpolation is performed in software.
         */
        CUDA_HOST void traverseAdjoint(cudaPitchedPtr& volume, const cudaPitchedPtr& sinogram);

        /**
         * @brief Approximation of backward projection for Josephs's method
         *
         * @param[in] volumeDims specifies the dimensions of the volume
         * @param[in] threads specifies the number of threads for each block
         * @param[out] volume pointer to output
         * @param[in] volumePitch pitch (padded width in bytes) of volume
         * @param[in] sinogram handle to the texture object containing the volume data
         * @param[in] rayOrigins pointer to ray origins
         * @param[in] originPitch pitch of ray origins
         * @param[in] proj pointer to projection matrices
         * @param[in] projPitch pitch of projection matrices
         * @param[in] numAngles number of acquisition angles
         *
         * volumeDims should be given as (volSizeZ (1 for 2D), volSizeY, volSizeX).
         *
         * threads should ideally be a multiple of the warp size (32 for all current GPUs).
         *
         * @warning Interpolation mode for texture should be set to cudaFilterModePoint when working
         * with doubles.
         */
        CUDA_HOST void traverseAdjointFast(cudaPitchedPtr& volume, cudaTextureObject_t sinogram);

    private:
        BoundingBoxCUDA<dim> _volumeBoundingBox;
        BoundingBoxCUDA<dim> _sinogramBoundingBox;
        dim3 _threadsPerBlock{2, 2, 8};

        /// inverse of of projection matrices; stored column-wise on GPU
        cudaPitchedPtr _projInvMatrices;

        /// projection matrices; stored column-wise on GPU
        cudaPitchedPtr _projMatrices;

        /// ray origins for each acquisition angle
        cudaPitchedPtr _rayOrigins;
    };
} // namespace elsa