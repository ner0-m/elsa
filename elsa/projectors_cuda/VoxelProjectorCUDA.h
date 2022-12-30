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

#include "XrayProjector.h"

#include "ProjectVoxelsCUDA.cuh"
#include "Luts.hpp"

namespace elsa
{
    template <typename data_t = real_t>
    class VoxelProjectorCUDA;

    template <typename data_t>
    struct XrayProjectorInnerTypes<VoxelProjectorCUDA<data_t>> {
        using value_type = data_t;
        using forward_tag = any_projection_tag;
        using backward_tag = any_projection_tag;
    };

    /**
     * @brief GPU-operator representing the discretized X-ray transform in 2d/3d using Voxel
     * Projection
     *
     * @author Noah Dormann
     *
     * @cite based on SiddonsMethodCUDA
     *
     * @tparam data_t data type for the domain and range of the operator, defaulting to real_t
     *
     * Every voxel in the volume is projected to the detector according to the Geometry and each hit
     * detector pixel is added the weighted voxel value.
     *
     * The geometry is represented as a list of projection matrices (see class Geometry), one for
     * each acquisition pose.
     *
     * Forward projection is accomplished using apply(), backward projection using applyAdjoint().
     *
     * Currently only utilizes a single GPU. Volume and images should both fit in device memory at
     * the same time.
     */
    template <typename data_t>
    class VoxelProjectorCUDA : public XrayProjector<VoxelProjectorCUDA<data_t>>
    {
    public:
        using self_type = VoxelProjectorCUDA<data_t>;
        using base_type = XrayProjector<self_type>;
        using value_type = typename base_type::value_type;
        using forward_tag = typename base_type::forward_tag;
        using backward_tag = typename base_type::backward_tag;

        /**
         * @brief Constructor for Voxel Projection.
         *
         * @param[in] radius the blob radius
         * @param[in] alpha blob hyperparameter
         * @param[in] order blob order
         * @param[in] domainDescriptor describing the domain of the operator (the volume)
         * @param[in] rangeDescriptor describing the range of the operator (the sinogram)
         *
         * The domain is expected to be 2 or 3 dimensional (volSizeX, volSizeY, [volSizeZ]),
         * the range is expected to be matching the domain (detSizeX, [detSizeY], acqPoses).
         */
        VoxelProjectorCUDA(data_t radius, data_t alpha, data_t order,
                           const VolumeDescriptor& domainDescriptor,
                           const DetectorDescriptor& rangeDescriptor);

        /**
         * @brief Constructor for Voxel Projection.
         *
         * @param[in] domainDescriptor describing the domain of the operator (the volume)
         * @param[in] rangeDescriptor describing the range of the operator (the sinogram)
         *
         * The domain is expected to be 2 or 3 dimensional (volSizeX, volSizeY, [volSizeZ]),
         * the range is expected to be matching the domain (detSizeX, [detSizeY], acqPoses).
         */
        VoxelProjectorCUDA(const VolumeDescriptor& domainDescriptor,
                           const DetectorDescriptor& rangeDescriptor);

        /// destructor
        ~VoxelProjectorCUDA() override;

    protected:
        /// default copy constructor, hidden from non-derived classes to prevent potential slicing
        VoxelProjectorCUDA(const VoxelProjectorCUDA<data_t>&) = default;

        /// apply Siddon's method (i.e. forward projection)
        void forward(const BoundingBox& aabb, const DataContainer<data_t>& x,
                     DataContainer<data_t>& Ax) const;

        /// apply the adjoint of Siddon's method (i.e. backward projection)
        void backward(const BoundingBox& aabb, const DataContainer<data_t>& y,
                      DataContainer<data_t>& Aty) const;

        /// implement the polymorphic clone operation
        VoxelProjectorCUDA<data_t>* _cloneImpl() const;

        /// implement the polymorphic comparison operation
        bool _isEqual(const LinearOperator<data_t>& other) const;

    private:
        /// threads per dimension used in the kernel execution configuration
        static const unsigned int THREADS_PER_BLOCK =
            ProjectVoxelsCUDA<data_t>::MAX_THREADS_PER_BLOCK;

        /// Reference to DetectorDescriptor stored in LinearOperator
        DetectorDescriptor& _detectorDescriptor;

        /// Reference to VolumeDescriptor stored in LinearOperator
        VolumeDescriptor& _volumeDescriptor;

        ProjectedBlobLut<real_t, 100> _lut;

        /// projection matrices; stored column-wise on GPU
        cudaPitchedPtr _projMatrices;

        /// extrinsic matrices; stored column-wise on GPU
        cudaPitchedPtr _extMatrices;

        /// lut array; stored on GPU
        thrust::device_vector<data_t> _lutArray;

        template <bool adjoint>
        void projectVoxels(const DataContainer<data_t>& volume,
                           const DataContainer<data_t>& sino) const;

        /// lift from base class
        using LinearOperator<data_t>::_domainDescriptor;
        using LinearOperator<data_t>::_rangeDescriptor;

        friend class XrayProjector<self_type>;
    };
} // namespace elsa
