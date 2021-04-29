#pragma once

#include "elsaDefines.h"
#include "Optimizer.h"

#ifdef ELSA_HAS_CUDNN_BACKEND
#include "CudnnMemory.h"
#endif

namespace elsa
{
    namespace ml
    {
        namespace detail
        {
            template <typename data_t>
            class OptimizerAdamImpl<data_t, MlBackend::Dnnl> : public OptimizerImplBase<data_t>
            {
            public:
                OptimizerAdamImpl(index_t size, data_t learningRate = data_t(0.001),
                                  data_t beta1 = data_t(0.9), data_t beta2 = data_t(0.999),
                                  data_t epsilon = data_t(1e-7));

                void updateParameter(const data_t* gradient, int batchSize, data_t* param) override;

            private:
                /// \copydoc OptimizerImplBase::learningRate_
                using OptimizerImplBase<data_t>::learningRate_;

                /// \copydoc OptimizerImplBase::step_
                using OptimizerImplBase<data_t>::step_;

                /// \copydoc OptimizerImplBase::size_
                using OptimizerImplBase<data_t>::size_;

                /// exponential decay for 1st order momenta
                data_t beta1_;

                /// exponential decay for 2nd order momenta
                data_t beta2_;

                /// epsilon-value for numeric stability
                data_t epsilon_;

                /// 1st momentum
                Eigen::ArrayX<data_t> firstMomentum_;

                /// 2nd momentum
                Eigen::ArrayX<data_t> secondMomentum_;
            };

            template <typename data_t>
            struct OptimizerFactory<data_t, MlBackend::Dnnl> {
                static std::shared_ptr<OptimizerImplBase<data_t>> run(Optimizer<data_t>* opt,
                                                                      index_t size)
                {
                    switch (opt->getOptimizerType()) {
                        case OptimizerType::Adam: {
                            auto downcastedOpt = dynamic_cast<Adam<data_t>*>(opt);
                            return std::make_shared<OptimizerAdamImpl<data_t, MlBackend::Dnnl>>(
                                size, downcastedOpt->getLearningRate(), downcastedOpt->getBeta1(),
                                downcastedOpt->getBeta2(), downcastedOpt->getEpsilon());
                        }
                        default:
                            assert(false && "This execution path should never be reached");
                    }
                    return nullptr;
                }
            };
        } // namespace detail
    }     // namespace ml
} // namespace elsa