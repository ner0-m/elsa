#pragma once

#include "DnnlLayer.h"

namespace elsa
{
    template <typename data_t>
    class DnnlLRNLayer : public DnnlLayer<data_t>
    {
    public:
        DnnlLRNLayer(const DataDescriptor& inputDescriptor, const DataDescriptor& outputDescriptor,
                     index_t localSize, data_t alpha, data_t beta, data_t k);

    private:
        using BaseType = DnnlLayer<data_t>;

        using DnnlMemory = typename BaseType::DnnlMemory;

        /// \copydoc DnnlLayer::compileForwardStream
        void compileForwardStream() override;

        /// \copydoc DnnlLayer::compileBackwardStream
        void compileBackwardStream() override;

        /// \copydoc DnnlLayer::_input
        using BaseType::_input;

        /// \copydoc DnnlLayer::_inputGradient
        using BaseType::_inputGradient;

        /// \copydoc DnnlLayer::_output
        using BaseType::_output;

        /// \copydoc DnnlLayer::_outputGradient
        using BaseType::_outputGradient;

        /// \copydoc DnnlLayer::_forwardStream
        using BaseType::_forwardStream;

        /// \copydoc DnnlLayer::_backwardStream
        using BaseType::_backwardStream;

        /// \copydoc DnnlLayer::_engine
        using BaseType::_engine;

        /// \copydoc DnnlLayer::_typeTag
        using BaseType::_typeTag;

        /// Local size for LRN
        dnnl::memory::dim _localSize;

        /// This layer's workspace memory
        DnnlMemory _workspace;

        /// LRN alpha parameter
        data_t _alpha;

        /// LRN beta parameter
        data_t _beta;

        /// LRN k parameter
        data_t _k;

        /// LRN forward primitive descriptor
        dnnl::lrn_forward::primitive_desc _forwardPrimitiveDescriptor;

        /// LRN backward primitive descriptor
        dnnl::lrn_backward::primitive_desc _backwardPrimitiveDescriptor;
    };
} // namespace elsa