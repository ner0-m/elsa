#pragma once

#include <cstring>

#include <cuda_runtime.h>

#include "elsa.h"
#include "LinearOperator.h"
#include "Geometry.h"
#include "BoundingBox.h"
#include "LogGuard.h"
#include "Timer.h"

#include "projector_kernels/TraverseJosephsCUDA.cuh"

namespace elsa
{   
    /**
     * \brief Wrapper for the CUDA projector based on Joseph's method
     * 
     * The projector provides two implementations for the backprojection - 
     * a precise backprojection, using the exact adjoint of the forward projection operator,
     * and a fast backprojection, which makes use of the GPU's hardware interpolation capabilities.
     * Currently only utilizes a single GPU.
     * Volume and images should both fit in device memory at the same time.
     * 
     * \author Nikola Dinev (nikola.dinev@tum.de)
     */
    template <typename data_t = real_t> 
    class JosephsMethodCUDA : public LinearOperator<data_t>
    {
    public:
        JosephsMethodCUDA() = delete;

        /**
         * \brief Construct a new projector for the angles defined in the geometry vector
         * 
         * \param domainDescriptor input descriptor
         * \param rangeDescriptor output descriptor
         * \param geom describes the projection angles
         * \param fast uses fast backprojection if set, otherwise precise
         * \param threadsPerBlock number of threads per block used for the kernel configuration
         * 
         */
        JosephsMethodCUDA(const DataDescriptor &domainDescriptor, const DataDescriptor &rangeDescriptor,
                     const std::vector<Geometry> &geometryList, bool fast = true);

        virtual ~JosephsMethodCUDA();

    protected:
        /// apply the binary method (i.e. forward projection)
        void _apply(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) override;

        /// apply the adjoint of the  binary method (i.e. backward projection)
        void _applyAdjoint(const DataContainer<data_t>& y, DataContainer<data_t>& Aty) override;

        /// implement the polymorphic clone operation
        JosephsMethodCUDA<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override;
        
    private:
        const BoundingBox _boundingBox;        
        const std::vector<Geometry> _geometryList;
        const int _threadsPerBlock = TraverseJosephsCUDA<data_t>::MAX_THREADS_PER_BLOCK;
        const bool _fast;

        // Inverse of of projection matrices; stored column-wise on GPU
        cudaPitchedPtr _projInvMatrices;

        // Projection matrices; stored column-wise on GPU
        cudaPitchedPtr _projMatrices;

        // Ray origin for each angle
        cudaPitchedPtr _rayOrigins;

        using cudaArrayFlags = unsigned int;

        /**
         * \brief Copies contents of a 3D data container to and from the gpu
         */
        template <cudaMemcpyKind direction, bool async=true>
        void copy3DDataContainer(void* hostData,const cudaPitchedPtr& gpuData, const cudaExtent& extent) const;

        /**
         * \brief Copies the entire contents of hostData to the GPU, returns the created texture object and its associated cudaArray
         */
        template <cudaArrayFlags flags = 0U>
        std::pair<cudaTextureObject_t, cudaArray*> copyTextureToGPU(const DataContainer<data_t>& hostData) const;

        /// lift from base class
        using LinearOperator<data_t>::_domainDescriptor;
        using LinearOperator<data_t>::_rangeDescriptor;
    };
}
