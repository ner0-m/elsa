#include "CudnnOptimizer.h"

namespace elsa
{
    namespace ml
    {
        namespace detail
        {
            /************************* OptimizerSGDImpl ***********************/

            template <typename data_t>
            OptimizerSGDImpl<data_t, MlBackend::Cudnn>::OptimizerSGDImpl(
                index_t size, std::shared_ptr<CudnnContext> cudnnContext, data_t learningRate,
                data_t momentum, bool nesterov)
                : OptimizerImplBase<data_t>(size, learningRate),
                  cudnnContext_(cudnnContext),
                  momentum_(momentum),
                  nesterov_(nesterov)
            {
                if (momentum_ != 0) {
                    velocity_ = std::make_shared<DeviceMemory<data_t>>(size_);
                    velocity_->fill(data_t(0));
                }
            }

            template <typename data_t>
            __global__ static void sgdVanillaKernel(index_t size, int batchSize,
                                                    data_t learningRate, const data_t* gradient,
                                                    data_t* param)
            {
                ELSA_CUDA_KERNEL_LOOP(index, size)
                {
                    param[index] -= learningRate * gradient[index] / static_cast<data_t>(batchSize);
                }
            }

            template <typename data_t>
            __global__ static void
                sgdNesterovKernel(index_t size, int batchSize, data_t learningRate, data_t momentum,
                                  data_t* velocity, const data_t* gradient, data_t* param)
            {
                ELSA_CUDA_KERNEL_LOOP(index, size)
                {
                    data_t gi = gradient[index] / static_cast<data_t>(batchSize);
                    data_t vi = momentum * velocity[index] - learningRate * gi;
                    param[index] = param[index] + momentum * vi - learningRate * gi;
                }
            }

            template <typename data_t>
            __global__ static void
                sgdMomentumKernel(index_t size, int batchSize, data_t learningRate, data_t momentum,
                                  data_t* velocity, const data_t* gradient, data_t* param)
            {
                ELSA_CUDA_KERNEL_LOOP(index, size)
                {
                    data_t gi = gradient[index] / static_cast<data_t>(batchSize);
                    data_t vi = momentum * velocity[index] - learningRate * gi;
                    param[index] += vi;
                }
            }
            template <typename data_t>
            void OptimizerSGDImpl<data_t, MlBackend::Cudnn>::updateParameter(const data_t* gradient,
                                                                             int batchSize,
                                                                             data_t* param)
            {
                if (momentum_ == 0) {
                    // vanilla stochastic gradient descent:
                    // param = param - learningRate * gradient
                    sgdVanillaKernel<<<ELSA_CUDA_GET_BLOCKS(size_), ELSA_CUDA_NUM_THREADS>>>(
                        size_, batchSize, learningRate_, gradient, param);
                } else {
                    assert(velocity_->getSize() == size_);
                    // nesterov momentum
                    if (nesterov_) {
                        sgdNesterovKernel<<<ELSA_CUDA_GET_BLOCKS(size_), ELSA_CUDA_NUM_THREADS>>>(
                            size_, batchSize, learningRate_, momentum_,
                            velocity_->getMemoryHandle(), gradient, param);
                    } else {
                        sgdMomentumKernel<<<ELSA_CUDA_GET_BLOCKS(size_), ELSA_CUDA_NUM_THREADS>>>(
                            size_, batchSize, learningRate_, momentum_,
                            velocity_->getMemoryHandle(), gradient, param);
                    }
                }
            }

            template class OptimizerSGDImpl<float, MlBackend::Cudnn>;

            /************************* OptimizerAdamImpl **********************/

            template <typename data_t>
            OptimizerAdamImpl<data_t, MlBackend::Cudnn>::OptimizerAdamImpl(
                index_t size, std::shared_ptr<CudnnContext> cudnnContext, data_t learningRate,
                data_t beta1, data_t beta2, data_t epsilon)
                : OptimizerImplBase<data_t>(size, learningRate),
                  beta1_(beta1),
                  beta2_(beta2),
                  epsilon_(epsilon),
                  cudnnContext_(cudnnContext)
            {
                firstMomentum_ = std::make_shared<DeviceMemory<data_t>>(size_);
                firstMomentum_->fill(data_t(0));

                secondMomentum_ = std::make_shared<DeviceMemory<data_t>>(size_);
                secondMomentum_->fill(data_t(0));
            }

            template <typename data_t>
            __global__ static void
                adamKernel(index_t size, int batchSize, data_t learningRate, data_t beta1,
                           data_t beta2, data_t epsilon, index_t step, data_t* firstMomentum,
                           data_t* secondMomentum, const data_t* gradient, data_t* param)
            {
                ELSA_CUDA_KERNEL_LOOP(index, size)
                {
                    data_t gi = gradient[index] / static_cast<data_t>(batchSize);

                    // update first momentum
                    data_t mi = firstMomentum[index] =
                        beta1 * firstMomentum[index] + (1 - beta1) * gi;

                    // second momentum
                    data_t vi = secondMomentum[index] =
                        beta2 * secondMomentum[index] + (1 - beta2) * gi * gi;

                    // update parameter
                    param[index] -= learningRate * mi / (sqrt(vi) + epsilon);
                }
            }

            template <typename data_t>
            void OptimizerAdamImpl<data_t, MlBackend::Cudnn>::updateParameter(
                const data_t* gradient, int batchSize, data_t* param)
            {
                assert(gradient != nullptr && param != nullptr);

                ++step_;

                // Invoke Adam kernel
                adamKernel<<<ELSA_CUDA_GET_BLOCKS(size_), ELSA_CUDA_NUM_THREADS>>>(
                    size_, batchSize, learningRate_, beta1_, beta2_, epsilon_, step_,
                    firstMomentum_->getMemoryHandle(), secondMomentum_->getMemoryHandle(), gradient,
                    param);
            }

            template class OptimizerAdamImpl<float, MlBackend::Cudnn>;
        } // namespace detail
    }     // namespace ml
} // namespace elsa