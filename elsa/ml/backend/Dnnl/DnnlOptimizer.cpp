#include "DnnlOptimizer.h"

namespace elsa::ml
{
    namespace detail
    {
        template <typename data_t>
        OptimizerAdamImpl<data_t, MlBackend::Dnnl>::OptimizerAdamImpl(index_t size,
                                                                      data_t learningRate,
                                                                      data_t beta1, data_t beta2,
                                                                      data_t epsilon)
            : OptimizerImplBase<data_t>(size, learningRate),
              beta1_(beta1),
              beta2_(beta2),
              epsilon_(epsilon)
        {
            firstMomentum_.setZero(size_);
            secondMomentum_.setZero(size_);
        }

        template <typename data_t>
        void OptimizerAdamImpl<data_t, MlBackend::Dnnl>::updateParameter(
            const data_t* gradient, [[maybe_unused]] int batchSize, data_t* param)
        {
            ++step_;

            Eigen::Map<const Eigen::ArrayX<data_t>> gradientMem(gradient, size_);
            Eigen::Map<Eigen::ArrayX<data_t>> paramMem(param, size_);

            // first momentum
            firstMomentum_ = beta1_ * firstMomentum_ + (1 - beta1_) * gradientMem;
            auto correctedFirstMomentum = firstMomentum_ / (1 - std::pow(beta1_, step_));

            // second momentum
            secondMomentum_ = beta2_ * secondMomentum_ + (1 - beta2_) * gradientMem * gradientMem;
            auto correctedSecondMomentum = secondMomentum_ / (1 - std::pow(beta2_, step_));

            paramMem = paramMem
                       - learningRate_ * correctedFirstMomentum
                             / (correctedSecondMomentum.sqrt() + epsilon_);
        }

        template class OptimizerAdamImpl<float, MlBackend::Dnnl>;
    } // namespace detail
} // namespace elsa::ml
