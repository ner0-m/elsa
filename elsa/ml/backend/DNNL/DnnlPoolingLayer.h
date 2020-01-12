#pragma once

#include "DnnlLayer.h"
#include "DataDescriptor.h"

#include "dnnl.hpp"

namespace elsa
{
    template <typename data_t>
    class DnnlPoolingLayer final : public DnnlLayer<data_t>
    {
    public:
        DnnlPoolingLayer(const DataDescriptor& inputDescriptor,
                         const DataDescriptor& outputDescriptor, const IndexVector_t& poolingWindow,
                         const IndexVector_t& poolingStride);

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

        dnnl::memory::dims _poolingStride;
        dnnl::memory::dims _poolingWindow;
        dnnl::memory::dims _poolingPadding;

        DnnlMemory _workspaceMemory;

        dnnl::pooling_forward::primitive_desc _forwardPrimitiveDescriptor;
        dnnl::pooling_backward::primitive_desc _backwardPrimitiveDescriptor;
    };
} // namespace elsa