#pragma once

#include <cstring>

#include <cuda_runtime.h>

#include "elsaDefines.h"
#include "LinearOperator.h"
#include "Geometry.h"
#include "BoundingBox.h"
#include "VolumeDescriptor.h"
#include "DetectorDescriptor.h"

#include "XrayProjector.h"

#include "TraverseJosephsCUDA.cuh"

namespace elsa
{
    template <typename data_t = real_t>
    class JosephsMethodCUDA;

    template <typename data_t>
    struct XrayProjectorInnerTypes<JosephsMethodCUDA<data_t>> {
        using value_type = data_t;
        using forward_tag = any_projection_tag;
        using backward_tag = any_projection_tag;
    };

    /**
     * @brief GPU-operator representing the discretized X-ray transform in 2d/3d using Joseph's
     * method.
     *
     * @author Nikola Dinev
     *
     * @tparam data_t data type for the domain and range of the operator, defaulting to real_t
     *
     * The volume is traversed along the rays as specified by the Geometry. For interior voxels
     * the sampling point is located in the middle of the two planes orthogonal to the main
     * direction of the ray. For boundary voxels the sampling point is located at the center of the
     * ray intersection with the voxel.
     *
     * The geometry is represented as a list of projection matrices (see class Geometry), one for
     * each acquisition pose.
     *
     * Forward projection is accomplished using apply(), backward projection using applyAdjoint().
     * The projector provides two implementations for the backward projection. The slow version is
     * matched, while the fast one is not.
     *
     * Currently only utilizes a single GPU. Volume and images should both fit in device memory at
     * the same time.
     *
     * @warning Hardware interpolation is only supported for JosephsMethodCUDA<float>
     * @warning Hardware interpolation is significantly less accurate than the software
     * interpolation
     */
    template <typename data_t>
    class JosephsMethodCUDA : public XrayProjector<JosephsMethodCUDA<data_t>>
    {
    public:
        using self_type = JosephsMethodCUDA<data_t>;
        using base_type = XrayProjector<self_type>;
        using value_type = typename base_type::value_type;
        using forward_tag = typename base_type::forward_tag;
        using backward_tag = typename base_type::backward_tag;

        /**
         * @brief Constructor for Joseph's traversal method.
         *
         * @param[in] domainDescriptor describing the domain of the operator (the volume)
         * @param[in] rangeDescriptor describing the range of the operator (the sinogram)
         * @param[in] fast performs fast backward projection if set, otherwise matched; forward
         * projection is unaffected
         *
         * The domain is expected to be 2 or 3 dimensional (volSizeX, volSizeY, [volSizeZ]),
         * the range is expected to be matching the domain (detSizeX, [detSizeY], acqPoses).
         */
        JosephsMethodCUDA(const VolumeDescriptor& domainDescriptor,
                          const DetectorDescriptor& rangeDescriptor, bool fast = true);

        /// destructor
        ~JosephsMethodCUDA() override;

    protected:
        /// copy constructor, used for cloning
        JosephsMethodCUDA(const JosephsMethodCUDA<data_t>& other);

        /// apply Siddon's method (i.e. forward projection)
        void forward(const BoundingBox& aabb, const DataContainer<data_t>& x,
                     DataContainer<data_t>& Ax) const;

        /// apply the adjoint of Siddon's method (i.e. backward projection)
        void backward(const BoundingBox& aabb, const DataContainer<data_t>& y,
                      DataContainer<data_t>& Aty) const;

        /// implement the polymorphic clone operation
        JosephsMethodCUDA<data_t>* _cloneImpl() const;

        /// implement the polymorphic comparison operation
        bool _isEqual(const LinearOperator<data_t>& other) const;

    private:
        /// Reference to DetectorDescriptor stored in LinearOperator
        DetectorDescriptor& _detectorDescriptor;

        /// Reference to VolumeDescriptor stored in LinearOperator
        VolumeDescriptor& _volumeDescriptor;

        /// threads per block used in the kernel execution configuration
        static const unsigned int THREADS_PER_BLOCK =
            TraverseJosephsCUDA<data_t>::MAX_THREADS_PER_BLOCK;

        /// flag specifying which version of the backward projection should be used
        const bool _fast;

        /// inverse of of projection matrices; stored column-wise on GPU
        cudaPitchedPtr _projInvMatrices;

        /// projection matrices; stored column-wise on GPU
        cudaPitchedPtr _projMatrices;

        /// ray origins for each acquisition angle
        cudaPitchedPtr _rayOrigins;

        /// convenience typedef for cuda array flags
        using cudaArrayFlags = unsigned int;

        enum class ContainerCpyKind { cpyContainerToRawGPU, cpyRawGPUToContainer };
        /**
         * @brief Copies contents of a 3D data container between GPU and host memory
         *
         * @tparam direction specifies the direction of the copy operation
         * @tparam async whether the copy should be performed asynchronously wrt. the host
         *
         * @param hostData pointer to host data
         * @param gpuData pointer to gpu data
         * @param[in] extent specifies the amount of data to be copied
         *
         * Note that hostData is expected to be a pointer to a linear memory region with no padding
         * between dimensions - e.g. the data in DataContainer is stored as a vector with no extra
         * padding, and the pointer to the start of the memory region can be retrieved as follows:
         *
         * DataContainer x;
         * void* hostData = (void*)&x[0];
         */
        template <ContainerCpyKind direction, bool async = true>
        void copy3DDataContainer(void* hostData, const cudaPitchedPtr& gpuData,
                                 const cudaExtent& extent) const;

        /**
         * @brief Copies the entire contents of DataContainer to the GPU texture memory
         *
         * @tparam cudaArrayFlags flags used for the creation of the cudaArray which will contain
         * the data
         *
         * @param[in] hostData the host data container
         *
         * @returns a pair of the created texture object and its associated cudaArray
         */
        template <cudaArrayFlags flags = 0U>
        std::pair<cudaTextureObject_t, cudaArray*>
            copyTextureToGPU(const DataContainer<data_t>& hostData) const;

        /// lift from base class
        using LinearOperator<data_t>::_domainDescriptor;
        using LinearOperator<data_t>::_rangeDescriptor;

        friend class XrayProjector<self_type>;
    };
} // namespace elsa
