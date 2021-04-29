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
        class DnnlFlattenLayer : public DnnlLayer<data_t>
        {
        public:
            using BaseType = DnnlLayer<data_t>;

            DnnlFlattenLayer(const VolumeDescriptor& inputDescriptor,
                             const VolumeDescriptor& outputDescriptor);

        private:
            void compileForwardStream() override;

            void compileBackwardStream() override;

            /// \copydoc DnnlTrainableLayer::_engine
            using BaseType::_engine;

            /// \copydoc DnnlTrainableLayer::_engine
            using BaseType::_forwardStream;

            /// \copydoc DnnlTrainableLayer::_input
            using BaseType::_input;

            /// \copydoc DnnlTrainableLayer::_inputGradient
            using BaseType::_inputGradient;

            /// \copydoc DnnlTrainableLayer::_output
            using BaseType::_output;

            /// \copydoc DnnlTrainableLayer::_typeTag
            using BaseType::_typeTag;

            /// \copydoc DnnlTrainableLayer::_outputGradient
            using BaseType::_outputGradient;
        };
    } // namespace detail
} // namespace elsa::ml