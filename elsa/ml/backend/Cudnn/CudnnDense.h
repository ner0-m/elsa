#pragma once

#include "CudnnTrainable.h"

namespace elsa
{
    namespace ml
    {
        namespace detail
        {
            template <typename data_t>
            class CudnnDense : public CudnnTrainable<data_t, /* isConvolution */ false>
            {
            public:
                CudnnDense(const VolumeDescriptor& inputDescriptor,
                           const VolumeDescriptor& outputDescriptor,
                           const VolumeDescriptor& weightsDescriptor, bool useBias,
                           Initializer weigthsInitializer, Initializer biasInitializer);

                ~CudnnDense();

                void compileForwardStream() override;

                void forwardPropagate() override;

                void backwardPropagate() override;

            private:
                using BaseType = CudnnTrainable<data_t>;

                using BaseType::cudnnContext_;

                using BaseType::input_;
                using BaseType::output_;
                using BaseType::outputGradient_;
                using BaseType::inputGradient_;

                using BaseType::weights_;
                using BaseType::weightsGradient_;
                using BaseType::bias_;
                using BaseType::biasGradient_;

                using BaseType::useBias_;

                data_t* vectorOfOnes_;
            };
        } // namespace detail
    }     // namespace ml
} // namespace elsa