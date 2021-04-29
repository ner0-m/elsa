#pragma once

#include "elsaDefines.h"
#include "CudnnLayer.h"
#include "Initializer.h"
#include "Optimizer.h"
#include "CudnnOptimizer.h"
#include "Common.h"

namespace elsa
{
    namespace ml
    {
        namespace detail
        {
            template <typename data_t, bool isConvolution = false>
            class CudnnTrainable : public CudnnLayer<data_t>
            {
            public:
                CudnnTrainable(const VolumeDescriptor& inputDescriptor,
                               const VolumeDescriptor& outputDescriptor,
                               const VolumeDescriptor& weightsDescriptor, bool useBias,
                               Initializer weigthsInitializer, Initializer biasInitializer);

                void updateTrainableParameters();

                void accumulateGradients();

                void initialize();

                bool isTrainable() const override;

                void setOptimizer(Optimizer<data_t>* optimizer);

                void compileForwardStream() override;

                void compileBackwardStream() override;

            protected:
                using BaseType = CudnnLayer<data_t>;

                using BaseType::cudnnContext_;

                CudnnMemory<data_t, /* isFilter */ isConvolution> weights_;
                CudnnMemory<data_t, /* isFilter */ isConvolution> weightsGradient_;
                CudnnMemory<data_t, /* isFilter */ isConvolution> weightsGradientAcc_;

                CudnnMemory<data_t, /* isFilter */ false> bias_;
                CudnnMemory<data_t, /* isFilter */ false> biasGradient_;
                CudnnMemory<data_t, /* isFilter */ false> biasGradientAcc_;

                /// This layer's initializer tag for weights
                Initializer weigthsInitializer_;

                /// This layer's initializer tag for bias
                Initializer biasInitializer_;

                bool useBias_;

                /// This layer's fanIn/fanOut pair that is used during random
                // initialization of weights and biases
                typename InitializerImpl<data_t>::FanPairType fanInOut_;

                std::shared_ptr<OptimizerImplBase<data_t>> weightsOptimizer_ = nullptr;
                std::shared_ptr<OptimizerImplBase<data_t>> biasOptimizer_ = nullptr;
            };
        } // namespace detail
    }     // namespace ml
} // namespace elsa