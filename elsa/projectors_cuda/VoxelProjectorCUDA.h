#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/universal_vector.h>

#include "elsaDefines.h"
#include "LinearOperator.h"
#include "Geometry.h"
#include "BoundingBox.h"
#include "VolumeDescriptor.h"
#include "DetectorDescriptor.h"
#include "Blobs.h"
#include "BSplines.h"
#include "XrayProjector.h"

#include "ProjectVoxelsCUDA.cuh"
#include "VoxelCUDAHelper.cuh"

namespace elsa
{

    template <typename data_t = float>
    class BlobVoxelProjectorCUDA : public LinearOperator<data_t>
    {
    public:
        /**
         * @brief Constructor for Blob Voxel Projection.
         *
         * @param[in] domainDescriptor describing the domain of the operator (the volume)
         * @param[in] rangeDescriptor describing the range of the operator (the sinogram)
         * @param[in] radius the blob radius
         * @param[in] alpha blob hyperparameter
         * @param[in] order blob order
         *
         * The domain is expected to be 2 or 3 dimensional (volSizeX, volSizeY, [volSizeZ]),
         * the range is expected to be matching the domain (detSizeX, [detSizeY], acqPoses).
         */
        BlobVoxelProjectorCUDA(const VolumeDescriptor& domainDescriptor,
                               const DetectorDescriptor& rangeDescriptor,
                               data_t radius = blobs::DEFAULT_RADIUS,
                               data_t alpha = blobs::DEFAULT_ALPHA,
                               index_t order = blobs::DEFAULT_ORDER);

        void applyImpl(const elsa::DataContainer<data_t>& x,
                       elsa::DataContainer<data_t>& Ax) const override;

        void applyAdjointImpl(const elsa::DataContainer<data_t>& y,
                              elsa::DataContainer<data_t>& Aty) const override;

        /// implement the polymorphic clone operation
        BlobVoxelProjectorCUDA<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override;

    public:
        ProjectedBlob<data_t> blob;

    private:
        index_t _dim;
        /// projection matrices; stored column-wise on GPU
        thrust::device_vector<data_t> _projMatrices;
        /// extrinsic matrices; stored column-wise on GPU
        thrust::device_vector<data_t> _extMatrices;
    };

    template <typename data_t = float>
    class BSplineVoxelProjectorCUDA : public LinearOperator<data_t>
    {
    public:
        /**
         * @brief Constructor for BSpline Voxel Projection.
         *
         * @param[in] domainDescriptor describing the domain of the operator (the volume)
         * @param[in] rangeDescriptor describing the range of the operator (the sinogram)
         * @param[in] order bspline order
         *
         * The domain is expected to be 2 or 3 dimensional (volSizeX, volSizeY, [volSizeZ]),
         * the range is expected to be matching the domain (detSizeX, [detSizeY], acqPoses).
         */
        BSplineVoxelProjectorCUDA(const VolumeDescriptor& domainDescriptor,
                                  const DetectorDescriptor& rangeDescriptor,
                                  index_t order = bspline::DEFAULT_ORDER);

        void applyImpl(const elsa::DataContainer<data_t>& x, elsa::DataContainer<data_t>& Ax) const;

        void applyAdjointImpl(const elsa::DataContainer<data_t>& y,
                              elsa::DataContainer<data_t>& Aty) const;

        /// implement the polymorphic clone operation
        BSplineVoxelProjectorCUDA<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override;

    public:
        ProjectedBSpline<data_t> bspline;

    private:
        index_t _dim;
        /// projection matrices; stored column-wise on GPU
        thrust::device_vector<data_t> _projMatrices;
        /// extrinsic matrices; stored column-wise on GPU
        thrust::device_vector<data_t> _extMatrices;
    };

    template <typename data_t = float>
    class PhaseContrastBlobVoxelProjectorCUDA : public LinearOperator<data_t>
    {
    public:
        /**
         * @brief Constructor for Phase Contrast Blob Voxel Projection.
         *
         * @param[in] domainDescriptor describing the domain of the operator (the volume)
         * @param[in] rangeDescriptor describing the range of the operator (the sinogram)
         * @param[in] radius the blob radius
         * @param[in] alpha blob hyperparameter
         * @param[in] order blob order
         *
         * The domain is expected to be 2 or 3 dimensional (volSizeX, volSizeY, [volSizeZ]),
         * the range is expected to be matching the domain (detSizeX, [detSizeY], acqPoses).
         */
        PhaseContrastBlobVoxelProjectorCUDA(const VolumeDescriptor& domainDescriptor,
                                            const DetectorDescriptor& rangeDescriptor,
                                            data_t radius = blobs::DEFAULT_RADIUS,
                                            data_t alpha = blobs::DEFAULT_ALPHA,
                                            index_t order = blobs::DEFAULT_ORDER);

        void applyImpl(const elsa::DataContainer<data_t>& x, elsa::DataContainer<data_t>& Ax) const;

        void applyAdjointImpl(const elsa::DataContainer<data_t>& y,
                              elsa::DataContainer<data_t>& Aty) const;

        /// implement the polymorphic clone operation
        PhaseContrastBlobVoxelProjectorCUDA<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override;

    public:
        ProjectedBlob<data_t> blob;

    private:
        index_t _dim;
        /// projection matrices; stored column-wise on GPU
        thrust::device_vector<data_t> _projMatrices;
        /// extrinsic matrices; stored column-wise on GPU
        thrust::device_vector<data_t> _extMatrices;
    };

    template <typename data_t = float>
    class PhaseContrastBSplineVoxelProjectorCUDA : public LinearOperator<data_t>
    {
    public:
        /**
         * @brief Constructor for Phase Contrast BSpline Voxel Projection.
         *
         * @param[in] domainDescriptor describing the domain of the operator (the volume)
         * @param[in] rangeDescriptor describing the range of the operator (the sinogram)
         * @param[in] order bspline order
         *
         * The domain is expected to be 2 or 3 dimensional (volSizeX, volSizeY, [volSizeZ]),
         * the range is expected to be matching the domain (detSizeX, [detSizeY], acqPoses).
         */
        PhaseContrastBSplineVoxelProjectorCUDA(const VolumeDescriptor& domainDescriptor,
                                               const DetectorDescriptor& rangeDescriptor,
                                               index_t order = bspline::DEFAULT_ORDER);

        void applyImpl(const elsa::DataContainer<data_t>& x,
                       elsa::DataContainer<data_t>& Ax) const override;

        void applyAdjointImpl(const elsa::DataContainer<data_t>& y,
                              elsa::DataContainer<data_t>& Aty) const override;

        /// implement the polymorphic clone operation
        PhaseContrastBSplineVoxelProjectorCUDA<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override;

    public:
        ProjectedBSpline<data_t> bspline;

    private:
        index_t _dim;
        /// projection matrices; stored column-wise on GPU
        thrust::device_vector<data_t> _projMatrices;
        /// extrinsic matrices; stored column-wise on GPU
        thrust::device_vector<data_t> _extMatrices;
    };
} // namespace elsa
