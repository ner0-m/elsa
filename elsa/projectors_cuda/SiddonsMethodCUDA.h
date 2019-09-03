#pragma once

#include <cuda_runtime.h>

#include "elsa.h"
#include "LinearOperator.h"
#include "Geometry.h"
#include "BoundingBox.h"
#include "LogGuard.h"
#include "Timer.h"

#include "projector_kernels/TraverseSiddonsCUDA.cuh"

namespace elsa
{
    /**
     * \brief Wrapper for the CUDA projector based on Siddon's method
     * 
     * Currently only utilizes a single GPU.
     * Volume and images should both fit in device memory at the same time.
     * 
     * \author Nikola Dinev (nikola.dinev@tum.de)
     */
    template <typename data_t = real_t>
    class SiddonsMethodCUDA : public LinearOperator<data_t>
    {
    public:
        SiddonsMethodCUDA() = delete;
        
        /**
         * \brief Construct a new projector for the angles defined in the geometry vector
         */
        SiddonsMethodCUDA(const DataDescriptor &domainDescriptor, const DataDescriptor &rangeDescriptor,
                     const std::vector<Geometry> &geom);


        ~SiddonsMethodCUDA();

    protected:
        /// apply the binary method (i.e. forward projection)
        void _apply(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) override;

        /// apply the adjoint of the  binary method (i.e. backward projection)
        void _applyAdjoint(const DataContainer<data_t>& y, DataContainer<data_t>& Aty) override;

        /// implement the polymorphic clone operation
        SiddonsMethodCUDA<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override;

    private:
        const BoundingBox _boundingBox;
        
        const std::vector<Geometry> _geometryList;
        const int _threadsPerBlock = 64;

        /**
         * Inverse of significant part of projection matrices; stored column-wise
         */
        cudaPitchedPtr _projInvMatrices;
        /**
         * Ray origin for each direction
         */
        cudaPitchedPtr _rayOrigins;

        uint* _boxMax;

        template<bool adjoint>
        void traverseVolume(void* volumePtr, void* sinoPtr) const;

        template <cudaMemcpyKind direction, bool async=true>
        void copy3DDataContainerGPU(void* hostData,const cudaPitchedPtr& gpuData, const cudaExtent& extent) const;

        /// lift from base class
        using LinearOperator<data_t>::_domainDescriptor;
        using LinearOperator<data_t>::_rangeDescriptor;
    };
}
