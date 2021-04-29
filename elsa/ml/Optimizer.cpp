#include "Optimizer.h"

namespace elsa
{
    namespace ml
    {
        template <typename data_t>
        Optimizer<data_t>::Optimizer(OptimizerType optimizerType, data_t learningRate)
            : optimizerType_(optimizerType), learningRate_(learningRate)
        {
        }

        template <typename data_t>
        OptimizerType Optimizer<data_t>::getOptimizerType() const
        {
            return optimizerType_;
        }

        template <typename data_t>
        SGD<data_t>::SGD(data_t learningRate, data_t momentum, bool nesterov)
            : Optimizer<data_t>(OptimizerType::SGD, learningRate),
              momentum_(momentum),
              nesterov_(nesterov)
        {
        }

        // template <typename data_t>
        // std::pair<Eigen::ArrayX<data_t>, Eigen::ArrayX<data_t>>
        //     SGD<data_t>::getParameterUpdates(const Eigen::ArrayX<data_t>& weights,
        //                                      const Eigen::ArrayX<data_t>& bias)
        // {
        //     if (!isInitialized_) {
        //         weightsVelocity_.setZero(weights.size());
        //         biasVelocity_.setZero(bias.size());
        //         isInitialized_ = true;
        //     }

        //     // Do we have momentum
        //     if (std::abs(momentum_) <= 0) {
        //         // Do we use Nesterov-momentum?
        //         if (nesterov_) {
        //             weightsVelocity_ = momentum_ * weightsVelocity_ - learningRate_ * weights;
        //             biasVelocity_ = momentum_ * biasVelocity_ - learningRate_ * bias;
        //             return std::make_pair<Eigen::ArrayX<data_t>, Eigen::ArrayX<data_t>>(
        //                 data_t(-1) * momentum_ * weightsVelocity_ - learningRate_ * weights,
        //                 data_t(-1) * momentum_ * weightsVelocity_ - learningRate_ * bias);
        //         } else {
        //             return std::make_pair<Eigen::ArrayX<data_t>, Eigen::ArrayX<data_t>>(
        //                 data_t(-1) * momentum_ * weightsVelocity_ + learningRate_ * weights,
        //                 data_t(-1) * momentum_ * weightsVelocity_ + learningRate_ * bias);
        //         }
        //     } else {
        //         // vanilla stochastic gradient descent
        //         return std::make_pair<Eigen::ArrayX<data_t>, Eigen::ArrayX<data_t>>(
        //             learningRate_ * weights, learningRate_ * bias);
        //     }
        // }

        template <typename data_t>
        data_t SGD<data_t>::getMomentum() const
        {
            return momentum_;
        }

        template <typename data_t>
        bool SGD<data_t>::useNesterov() const
        {
            return nesterov_;
        }

        template <typename data_t>
        Adam<data_t>::Adam(data_t learningRate, data_t beta1, data_t beta2, data_t epsilon)
            : Optimizer<data_t>(OptimizerType::Adam, learningRate),
              beta1_(beta1),
              beta2_(beta2),
              epsilon_(epsilon)
        {
        }

        template <typename data_t>
        data_t Adam<data_t>::getBeta1() const
        {
            return beta1_;
        }

        template <typename data_t>
        data_t Adam<data_t>::getBeta2() const
        {
            return beta2_;
        }

        template <typename data_t>
        data_t Adam<data_t>::getEpsilon() const
        {
            return epsilon_;
        }

        template class Adam<float>;
        template class SGD<float>;
    } // namespace ml
} // namespace elsa