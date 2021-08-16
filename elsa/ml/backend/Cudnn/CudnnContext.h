#pragma once

#include "cublas_v2.h"
#include "cudnn.h"

#include "elsaDefines.h"

namespace elsa
{
    namespace ml
    {
        namespace detail
        {
            /// Context in which a Cudnn backend layer is executed. This
            /// includes handles for executin Cudnn and Cublas functions.
            class CudnnContext
            {
            public:
                /// Construct a Cudnn context. This constructs a Cudnn and a
                /// Cublas handle.
                CudnnContext();

                /// Destruct a Cudnn context. This destructs the Cudnn and the
                /// Cublas handle.
                ~CudnnContext();

                /// Get a reference to the Cublas handle
                cublasHandle_t& getCublasHandle();

                /// Get a constant reference to the Cublas handle
                const cublasHandle_t& getCublasHandle() const;

                /// Get a reference to the Cudnn handle
                cudnnHandle_t& getCudnnHandle();

                /// Get a constant reference to the Cudnn handle
                const cudnnHandle_t& getCudnnHandle() const;

                /// The constant float value 1.f
                static const float One;

                /// The constant float value 0.f
                static const float Zero;

                /// The constant float value -1.f
                static const float MinusOne;

            private:
                cublasHandle_t cublasHandle_;
                cudnnHandle_t cudnnHandle_;
            };
        } // namespace detail
    }     // namespace ml
} // namespace elsa