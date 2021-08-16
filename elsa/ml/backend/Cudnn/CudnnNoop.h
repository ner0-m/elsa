#pragma once

#include "elsaDefines.h"
#include "CudnnLayer.h"
#include "CudnnCommon.h"

namespace elsa
{
    namespace ml
    {
        namespace detail
        {
            /// A noop layer, i.e., a layer that does just nothing
            ///
            /// @author David Tellenbach
            ///
            /// This layer just forwards its input unchanged to its output and
            /// its output-gradient unchanged to its input-gradient.
            template <typename data_t>
            class CudnnNoop : public CudnnLayer<data_t>
            {
            public:
                /// Construct a CudnnNoop layer by providing an input descriptor
                CudnnNoop(const VolumeDescriptor& inputDescriptor);

                /// \copydoc CudnnLayer::forwardPropagate
                void forwardPropagate() override;

                /// \copydoc CudnnLayer::compileForwardStream
                void compileForwardStream() override;

                /// \copydoc CudnnLayer::backwardPropagate
                void backwardPropagate() override;

                /// \copydoc CudnnLayer::compileBackwardStream
                void compileBackwardStream() override;

            private:
                using BaseType = CudnnLayer<data_t>;

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
