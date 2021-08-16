#pragma once

#include "elsaDefines.h"
#include "CudnnLayer.h"
#include "CudnnNoop.h"
#include "CudnnDense.h"
#include "CudnnActivation.h"
#include "CudnnSoftmax.h"

namespace elsa
{
    namespace ml
    {
        namespace detail
        {
            template <typename data_t>
            class CudnnMerging : public CudnnLayer<data_t>
            {
            public:
                /// Construct a merging layer by specifying a list of input- and a single
                /// output-descriptor.
                CudnnMerging(const std::vector<VolumeDescriptor>& inputDescriptors,
                             const VolumeDescriptor& outputDescriptor);

                /// \copydoc CudnnLayer::needsForwardSynchronisation
                bool needsForwardSynchronisation() const override;

                /// \copydoc CudnnLayer::canMerge
                bool canMerge() const override;

            protected:
                using BaseType = CudnnLayer<data_t>;

                /// \copydoc CudnnLayer::cudnnContext_
                using BaseType::cudnnContext_;

                /// \copydoc CudnnLayer::input_
                using BaseType::input_;

                /// \copydoc CudnnLayer::inputGradient_
                using BaseType::inputGradient_;

                /// \copydoc CudnnLayer::output_
                using BaseType::output_;

                /// \copydoc CudnnLayer::outputGradient_
                using BaseType::outputGradient_;
            };

            template <typename data_t>
            class CudnnSum : public CudnnMerging<data_t>
            {
            public:
                /// Construct a sum layer by specifying a list of input- and a single
                /// output-descriptor.
                CudnnSum(const std::vector<VolumeDescriptor>& inputDescriptors,
                         const VolumeDescriptor& outputDescriptor);

                /// \copydoc CudnnLayer::forwardPropagate
                void forwardPropagate() override;

                /// \copydoc CudnnLayer::backwardPropagate
                void backwardPropagate() override;

            private:
                using BaseType = CudnnMerging<data_t>;

                /// \copydoc CudnnLayer::cudnnContext_
                using BaseType::cudnnContext_;

                /// \copydoc CudnnLayer::input_
                using BaseType::input_;

                /// \copydoc CudnnLayer::inputGradient_
                using BaseType::inputGradient_;

                /// \copydoc CudnnLayer::output_
                using BaseType::output_;

                /// \copydoc CudnnLayer::outputGradient_
                using BaseType::outputGradient_;
            };

            template <typename data_t>
            class CudnnConcatenate : public CudnnMerging<data_t>
            {
            public:
                /// Construct a concatenate layer by specifying a list of input- and a single
                /// output-descriptor.
                CudnnConcatenate(const std::vector<VolumeDescriptor>& inputDescriptors,
                                 const VolumeDescriptor& outputDescriptor);

                /// \copydoc CudnnLayer::forwardPropagate
                void forwardPropagate() override;

                /// \copydoc CudnnLayer::backwardPropagate
                void backwardPropagate() override;

            private:
                using BaseType = CudnnMerging<data_t>;

                /// \copydoc CudnnLayer::cudnnContext_
                using BaseType::cudnnContext_;

                /// \copydoc CudnnLayer::input_
                using BaseType::input_;

                /// \copydoc CudnnLayer::inputGradient_
                using BaseType::inputGradient_;

                /// \copydoc CudnnLayer::output_
                using BaseType::output_;

                /// \copydoc CudnnLayer::outputGradient_
                using BaseType::outputGradient_;
            };
        } // namespace detail
    }     // namespace ml
} // namespace elsa