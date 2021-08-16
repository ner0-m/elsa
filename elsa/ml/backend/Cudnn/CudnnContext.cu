#include "CudnnContext.h"

namespace elsa
{
    namespace ml
    {
        namespace detail
        {
            CudnnContext::CudnnContext()
            {
                cublasCreate(&cublasHandle_);
                cudnnCreate(&cudnnHandle_);
            }

            CudnnContext::~CudnnContext()
            {
                cublasDestroy(cublasHandle_);
                cudnnDestroy(cudnnHandle_);
            }

            cublasHandle_t& CudnnContext::getCublasHandle() { return cublasHandle_; }
            const cublasHandle_t& CudnnContext::getCublasHandle() const { return cublasHandle_; }

            cudnnHandle_t& CudnnContext::getCudnnHandle() { return cudnnHandle_; }
            const cudnnHandle_t& CudnnContext::getCudnnHandle() const { return cudnnHandle_; }

            const float CudnnContext::One = 1.f;
            const float CudnnContext::Zero = 0.f;
            const float CudnnContext::MinusOne = -1.f;

        } // namespace detail
    }     // namespace ml
} // namespace elsa