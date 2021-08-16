#pragma once

#include "elsaDefines.h"
#include "Common.h"
#include "DnnlLayer.h"
#include "VolumeDescriptor.h"

#include "dnnl.hpp"

namespace elsa::ml
{
    namespace detail
    {
        template <typename data_t>
        class DnnlNoopLayer : public DnnlLayer<data_t>
        {
        public:
            using BaseType = DnnlLayer<data_t>;

            explicit DnnlNoopLayer(const VolumeDescriptor& inputDescriptor);

        private:
            /// \copydoc DnnlLayer::compileForwardStream
            void compileForwardStream() override;

            /// \copydoc DnnlLayer::compileBackwardStream
            void compileBackwardStream() override;

            /// \copydoc DnnlLayer::_input
            using BaseType::_forwardStream;

            /// \copydoc DnnlLayer::_input
            using BaseType::_input;

            /// \copydoc DnnlLayer::_output
            using BaseType::_output;

            /// \copydoc DnnlLayer::_inputGradient
            using BaseType::_inputGradient;

            /// \copydoc DnnlLayer::_outputGradient
            using BaseType::_outputGradient;
        };
    } // namespace detail
} // namespace elsa::ml