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

#include "TraverseSiddonsCUDA.cuh"

namespace elsa
{
    template <typename data_t = real_t>
    class SiddonsMethodCUDA;

    template <typename data_t>
    struct XrayProjectorInnerTypes<SiddonsMethodCUDA<data_t>> {
        using value_type = data_t;
        using forward_tag = any_projection_tag;
        using backward_tag = any_projection_tag;
    };

    /**
     * @brief GPU-operator representing the discretized X-ray transform in 2d/3d using Siddon's
     * method.
     *
     * @author Nikola Dinev
     *
     * @tparam data_t data type for the domain and range of the operator, defaulting to real_t
     *
     * The volume is traversed along the rays as specified by the Geometry. Each ray is traversed in
     * a contiguous fashion (i.e. along long voxel borders, not diagonally) and each traversed
     * voxel is counted as a hit with weight according to the length of the path of the ray through
     * the voxel.
     *
     * The geometry is represented as a list of projection matrices (see class Geometry), one for
     * each acquisition pose.
     *
     * Forward projection is accomplished using apply(), backward projection using applyAdjoint().
     * This projector is matched.
     *
     * Currently only utilizes a single GPU. Volume and images should both fit in device memory at
     * the same time.
     */
    template <typename data_t>
    class SiddonsMethodCUDA : public XrayProjector<SiddonsMethodCUDA<data_t>>
    {
    public:
        using self_type = SiddonsMethodCUDA<data_t>;
        using base_type = XrayProjector<self_type>;
        using value_type = typename base_type::value_type;
        using forward_tag = typename base_type::forward_tag;
        using backward_tag = typename base_type::backward_tag;

        /**
         * @brief Constructor for Siddon's method traversal.
         *
         * @param[in] domainDescriptor describing the domain of the operator (the volume)
         * @param[in] rangeDescriptor describing the range of the operator (the sinogram)
         *
         * The domain is expected to be 2 or 3 dimensional (volSizeX, volSizeY, [volSizeZ]),
         * the range is expected to be matching the domain (detSizeX, [detSizeY], acqPoses).
         */
        SiddonsMethodCUDA(const VolumeDescriptor& domainDescriptor,
                          const DetectorDescriptor& rangeDescriptor);

        /// destructor
        ~SiddonsMethodCUDA() override = default;

    protected:
        /// default copy constructor, hidden from non-derived classes to prevent potential slicing
        SiddonsMethodCUDA(const SiddonsMethodCUDA<data_t>&) = default;

        /// apply Siddon's method (i.e. forward projection)
        void forward(const BoundingBox& aabb, const DataContainer<data_t>& x,
                     DataContainer<data_t>& Ax) const;

        /// apply the adjoint of Siddon's method (i.e. backward projection)
        void backward(const BoundingBox& aabb, const DataContainer<data_t>& y,
                      DataContainer<data_t>& Aty) const;

        /// implement the polymorphic clone operation
        SiddonsMethodCUDA<data_t>* _cloneImpl() const;

        /// implement the polymorphic comparison operation
        bool _isEqual(const LinearOperator<data_t>& other) const;

    private:
        /// threads per block used in the kernel execution configuration
        static const unsigned int THREADS_PER_BLOCK =
            TraverseSiddonsCUDA<data_t>::MAX_THREADS_PER_BLOCK;

        /// inverse of of projection matrices; stored column-wise on GPU
        // cudaPitchedPtr _projInvMatrices;
        thrust::device_vector<real_t> _invProjMatrices;

        /// ray origins for each acquisition angle, TODO: enable kernel to have this as data_t
        thrust::device_vector<real_t> _rayOrigins;

        void traverseForward(const BoundingBox& aabb,
                             const thrust::universal_vector<data_t>& volume,
                             thrust::universal_vector<data_t>& sino) const;

        void traverseBackward(const BoundingBox& aabb, thrust::universal_vector<data_t>& volume,
                              const thrust::universal_vector<data_t>& sino) const;

        /// lift from base class
        using LinearOperator<data_t>::_domainDescriptor;
        using LinearOperator<data_t>::_rangeDescriptor;

        friend class XrayProjector<self_type>;
    };
} // namespace elsa
