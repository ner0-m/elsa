#pragma once

#include "elsaDefines.h"
#include "Optimizer.h"
#include "CudnnMemory.h"
#include "CudnnContext.h"

namespace elsa
{
    namespace ml
    {
        namespace detail
        {
            template <typename data_t>
            class OptimizerSGDImpl<data_t, MlBackend::Cudnn> : public OptimizerImplBase<data_t>
            {
            public:
                OptimizerSGDImpl(index_t size, std::shared_ptr<CudnnContext> cudnnContext,
                                 data_t learningRate = data_t(0.01), data_t momentum = data_t(0.0),
                                 bool nesterov = false);

                void updateParameter(const data_t* gradient, index_t batchSize,
                                     data_t* param) override;

            private:
                /// \copydoc OptimizerImplBase::learningRate_
                using OptimizerImplBase<data_t>::learningRate_;

                /// \copydoc OptimizerImplBase::step_
                using OptimizerImplBase<data_t>::size_;

                data_t momentum_;

                bool nesterov_;

                /// velocity
                std::shared_ptr<DeviceMemory<data_t>> velocity_ = nullptr;

                std::shared_ptr<CudnnContext> cudnnContext_;
            };

            template <typename data_t>
            class OptimizerAdamImpl<data_t, MlBackend::Cudnn> : public OptimizerImplBase<data_t>
            {
            public:
                OptimizerAdamImpl(index_t size, std::shared_ptr<CudnnContext> cudnnContext,
                                  data_t learningRate = data_t(0.001), data_t beta1 = data_t(0.9),
                                  data_t beta2 = data_t(0.999), data_t epsilon = data_t(1e-7));

                void updateParameter(const data_t* gradient, int batchSize, data_t* param) override;

            private:
                /// \copydoc OptimizerImplBase::size_
                using OptimizerImplBase<data_t>::size_;

                /// \copydoc OptimizerImplBase::step_
                using OptimizerImplBase<data_t>::step_;

                /// \copydoc OptimizerImplBase::learningRate_
                using OptimizerImplBase<data_t>::learningRate_;

                /// exponential decay for 1st order momenta
                data_t beta1_;

                /// exponential decay for 2nd order momenta
                data_t beta2_;

                /// epsilon-value for numeric stability
                data_t epsilon_;

                /// 1st momentum
                std::shared_ptr<DeviceMemory<data_t>> firstMomentum_ = nullptr;

                /// 2nd momentum
                std::shared_ptr<DeviceMemory<data_t>> secondMomentum_ = nullptr;

                std::shared_ptr<CudnnContext> cudnnContext_;
            };

            template <typename data_t>
            struct OptimizerFactory<data_t, MlBackend::Cudnn> {
                static std::shared_ptr<OptimizerImplBase<data_t>>
                    run(Optimizer<data_t>* opt, index_t size, std::shared_ptr<CudnnContext> context)
                {
                    switch (opt->getOptimizerType()) {
                        case OptimizerType::Adam: {
                            auto downcastedOpt = downcast<Adam<data_t>>(opt);
                            return std::make_shared<OptimizerAdamImpl<data_t, MlBackend::Cudnn>>(
                                size, context, downcastedOpt->getLearningRate(),
                                downcastedOpt->getBeta1(), downcastedOpt->getBeta2(),
                                downcastedOpt->getEpsilon());
                        }
                        case OptimizerType::SGD: {
                            auto downcastedOpt = downcast<SGD<data_t>>(opt);
                            return std::make_shared<OptimizerSGDImpl<data_t, MlBackend::Cudnn>>(
                                size, context, downcastedOpt->getLearningRate(),
                                downcastedOpt->getMomentum(), downcastedOpt->useNesterov());
                        }
                        default:
                            assert(false
                                   && "This execution path of the code should never be taken.");
                    }
                    return nullptr;
                }
            };
        } // namespace detail
    }     // namespace ml
} // namespace elsa
