#pragma once

#include <cuda_runtime.h>

#include "elsaDefines.h"
#include "LinearOperator.h"
#include "Geometry.h"
#include "BoundingBox.h"

#include "TraverseSiddonsCUDA.cuh"

namespace elsa
{
    /**
     * \brief GPU-operator representing the discretized X-ray transform in 2d/3d using Siddon's
     * method.
     *
     * \author Nikola Dinev
     *
     * \tparam data_t data type for the domain and range of the operator, defaulting to real_t
     *
     * The volume is traversed along the rays as specified by the Geometry. Each ray is traversed in
     * a continguous fashion (i.e. along long voxel borders, not diagonally) and each traversed
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
    template <typename data_t = real_t>
    class SiddonsMethodCUDA : public LinearOperator<data_t>
    {
    public:
        /**
         * \brief Constructor for Siddon's method traversal.
         *
         * \param[in] domainDescriptor describing the domain of the operator (the volume)
         * \param[in] rangeDescriptor describing the range of the operator (the sinogram)
         * \param[in] geometryList vector containing the geometries for the acquisition poses
         *
         * The domain is expected to be 2 or 3 dimensional (volSizeX, volSizeY, [volSizeZ]),
         * the range is expected to be matching the domain (detSizeX, [detSizeY], acqPoses).
         */
        SiddonsMethodCUDA(const DataDescriptor& domainDescriptor,
                          const DataDescriptor& rangeDescriptor,
                          const std::vector<Geometry>& geometryList);

        /// destructor
        ~SiddonsMethodCUDA();

    protected:
        /// default copy constructor, hidden from non-derived classes to prevent potential slicing
        SiddonsMethodCUDA(const SiddonsMethodCUDA<data_t>&) = default;

        /// apply Siddon's method (i.e. forward projection)
        void applyImpl(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const override;

        /// apply the adjoint of Siddon's method (i.e. backward projection)
        void applyAdjointImpl(const DataContainer<data_t>& y,
                              DataContainer<data_t>& Aty) const override;

        /// implement the polymorphic clone operation
        SiddonsMethodCUDA<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override;

    private:
        /// the bounding box of the volume
        BoundingBox _boundingBox;

        /// the geometry list
        std::vector<Geometry> _geometryList;

        /// threads per block used in the kernel execution configuration
        const int _threadsPerBlock = TraverseSiddonsCUDA<data_t>::MAX_THREADS_PER_BLOCK;

        /// inverse of of projection matrices; stored column-wise on GPU
        cudaPitchedPtr _projInvMatrices;

        /// ray origins for each acquisition angle
        cudaPitchedPtr _rayOrigins;

        /// sets up and starts the kernel traversal routine (for both apply/applyAdjoint)
        template <bool adjoint>
        void traverseVolume(void* volumePtr, void* sinoPtr) const;

        /**
         * \brief Copies contents of a 3D data container between GPU and host memory
         *
         * \tparam direction specifies the direction of the copy operation
         * \tparam async whether the copy should be performed asynchronously wrt. the host
         *
         * \param hostData pointer to host data
         * \param gpuData pointer to gpu data
         * \param[in] extent specifies the amount of data to be copied
         *
         * Note that hostData is expected to be a pointer to a linear memory region with no padding
         * between dimensions - e.g. the data in DataContainer is stored as a vector with no extra
         * padding, and the pointer to the start of the memory region can be retrieved as follows:
         *
         * DataContainer x;
         * void* hostData = (void*)&x[0];
         */
        template <cudaMemcpyKind direction, bool async = true>
        void copy3DDataContainerGPU(void* hostData, const cudaPitchedPtr& gpuData,
                                    const cudaExtent& extent) const;

        /// lift from base class
        using LinearOperator<data_t>::_domainDescriptor;
        using LinearOperator<data_t>::_rangeDescriptor;
    };
} // namespace elsa
