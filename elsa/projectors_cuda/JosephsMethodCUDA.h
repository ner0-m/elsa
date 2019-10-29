#pragma once

#include <cstring>

#include <cuda_runtime.h>

#include "elsa.h"
#include "LinearOperator.h"
#include "Geometry.h"
#include "BoundingBox.h"

#include "TraverseJosephsCUDA.cuh"

namespace elsa
{   
    /**
     * \brief GPU-operator representing the discretized X-ray transform in 2d/3d using Joseph's method.
     *
     * \author Nikola Dinev
     * 
     * \tparam data_t data type for the domain and range of the operator, defaulting to real_t
     * 
     * The volume is traversed along the rays as specified by the Geometry. For interior voxels
     * the sampling point is located in the middle of the two planes orthogonal to the main
     * direction of the ray. For boundary voxels the sampling point is located at the center of the
     * ray intersection with the voxel.
     * 
     * The geometry is represented as a list of projection matrices (see class Geometry), one for each
     * acquisition pose.
     * 
     * Forward projection is accomplished using apply(), backward projection using applyAdjoint().
     * The projector provides two implementations for the backward projection. The slow version is matched,
     * while the fast one is not. 
     * 
     * Currently only utilizes a single GPU. Volume and images should both fit in device memory at the same time.
     * 
     * \warning Hardware interpolation is only supported for JosephsMethodCUDA<float>
     * \warning Hardware interpolation is significantly less accurate than the software interpolation
     */
    template <typename data_t = real_t> 
    class JosephsMethodCUDA : public LinearOperator<data_t>
    {
    public:

        /**
         * \brief Constructor for Joseph's traversal method.
         * 
         * \param[in] domainDescriptor describing the domain of the operator (the volume)
         * \param[in] rangeDescriptor describing the range of the operator (the sinogram)
         * \param[in] geometryList vector containing the geometries for the acquisition poses
         * \param[in] fast performs fast backward projection if set, otherwise matched; forward projection is unaffected
         * 
         * The domain is expected to be 2 or 3 dimensional (volSizeX, volSizeY, [volSizeZ]),
         * the range is expected to be matching the domain (detSizeX, [detSizeY], acqPoses).
         */
        JosephsMethodCUDA(const DataDescriptor &domainDescriptor, const DataDescriptor &rangeDescriptor,
                     const std::vector<Geometry> &geometryList, bool fast = true);

        /// destructor
        virtual ~JosephsMethodCUDA();

    protected:
        /// apply Joseph's method (i.e. forward projection)
        void _apply(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const override;

        /// apply the adjoint of Joseph's method (i.e. backward projection)
        void _applyAdjoint(const DataContainer<data_t>& y, DataContainer<data_t>& Aty) const override;

        /// implement the polymorphic clone operation
        JosephsMethodCUDA<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override;
        
    private:
        /// the bounding box of the volume
        BoundingBox _boundingBox;

        /// the geometry list
        std::vector<Geometry> _geometryList;

        /// threads per block used in the kernel execution configuration
        const int _threadsPerBlock = TraverseJosephsCUDA<data_t>::MAX_THREADS_PER_BLOCK;

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
         * Note that hostData is expected to be a pointer to a linear memory region with no padding between
         * dimensions - e.g. the data in DataContainer is stored as a vector with no extra padding, and the 
         * pointer to the start of the memory region can be retrieved as follows:
         * 
         * DataContainer x;
         * void* hostData = (void*)&x[0];
         */
        template <cudaMemcpyKind direction, bool async=true>
        void copy3DDataContainer(void* hostData,const cudaPitchedPtr& gpuData, const cudaExtent& extent) const;

        /**
         * \brief Copies the entire contents of DataContainer to the GPU texture memory
         * 
         * \tparam cudaArrayFlags flags used for the creation of the cudaArray which will contain the data
         * 
         * \param[in] hostData the host data container
         * 
         * \returns a pair of the created texture object and its associated cudaArray
         */
        template <cudaArrayFlags flags = 0U>
        std::pair<cudaTextureObject_t, cudaArray*> copyTextureToGPU(const DataContainer<data_t>& hostData) const;

        /// lift from base class
        using LinearOperator<data_t>::_domainDescriptor;
        using LinearOperator<data_t>::_rangeDescriptor;
    };
}
