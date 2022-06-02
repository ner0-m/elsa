#pragma once

#include "TraverseJosephsCUDA.cuh"

#include "VolumeDescriptor.h"
#include "DetectorDescriptor.h"
#include "CUDAProjector.h"

#include <cstring>
#include <optional>

namespace elsa
{
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
    template <typename data_t = real_t>
    class JosephsMethodCUDA : public CUDAProjector<data_t>
    {
    public:
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
        ~JosephsMethodCUDA() override = default;

        BoundingBox constrainProjectionSpace(const BoundingBox& aabb) const override;

        BoundingBox constrainProjectionSpace2(const BoundingBox& imagePatch,
                                              const std::vector<Interval>& poses) const;

        std::unique_ptr<CUDAVariablesForward<data_t>>
            setupCUDAVariablesForward(IndexVector_t chunkSizeDomain,
                                      IndexVector_t chunkSizeRange) const override;

        void applyConstrained(const DataContainer<data_t>& x, DataContainer<data_t>& Ax,
                              const ForwardProjectionTask& task,
                              CUDAVariablesForward<data_t>& cudaVars) const override;

        std::vector<ForwardProjectionTask>
            getSubtasks(const ForwardProjectionTask& task, const IndexVector_t& maxVolumeDims,
                        const IndexVector_t& maxImageDims) const override;

    protected:
        /// copy constructor, used for cloning
        JosephsMethodCUDA(const JosephsMethodCUDA<data_t>& other);

        void copyDataForward(const data_t* x, const BoundingBox& volumeBox,
                             const CUDAVariablesForward<data_t>& cudaVars) const;

        /// apply Joseph's method (i.e. forward projection)
        void applyImpl(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const override;

        /// apply the adjoint of Joseph's method (i.e. backward projection)
        void applyAdjointImpl(const DataContainer<data_t>& y,
                              DataContainer<data_t>& Aty) const override;

        /// implement the polymorphic clone operation
        JosephsMethodCUDA<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override;

    private:
        static constexpr index_t PREFERRED_AOR = 2;

        std::optional<index_t> _aor;

        /// threads per block used in the kernel execution configuration
        static const unsigned int THREADS_PER_BLOCK =
            TraverseJosephsCUDA<data_t>::MAX_THREADS_PER_BLOCK;

        /// object describing the 2D projection problem in CUDA variables
        std::shared_ptr<TraverseJosephsCUDA<data_t, 2>> _traverse2D;

        /// object describing the 3D projection problem in CUDA variables
        std::shared_ptr<TraverseJosephsCUDA<data_t, 3>> _traverse3D;

        /// flag specifying which version of the backward projection should be used
        const bool _fast;

        void retrieveResults(data_t* hostData, const cudaPitchedPtr& gpuData,
                             const BoundingBox& aabb, const cudaStream_t& stream) const;

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
                                 const cudaExtent& extent, const cudaStream_t& stream) const;

        /// convenience typedef for cuda array flags
        using cudaArrayFlags = unsigned int;

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
        std::pair<cudaTextureObject_t, cudaArray*> copyTextureToGPU(const void* hostData,
                                                                    const BoundingBox& aabb,
                                                                    cudaStream_t stream) const;

        template <unsigned int flags = 0U>
        void copyTextureToGPUForward(const void* hostData, const BoundingBox& aabb,
                                     cudaArray_t darray, cudaStream_t stream) const;

        /// lift from base class
        using CUDAProjector<data_t>::_volumeDescriptor;
        using CUDAProjector<data_t>::_detectorDescriptor;
        using CUDAProjector<data_t>::_device;
        using CUDAProjector<data_t>::containerChunkToPinned;
        using CUDAProjector<data_t>::pinnedToContainerChunks;
        using CUDAProjector<data_t>::posesToContainer;
    };
} // namespace elsa
