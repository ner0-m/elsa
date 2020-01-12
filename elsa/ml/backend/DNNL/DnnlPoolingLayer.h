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
        void compileForwardStream() override;
        void compileBackwardStream() override;

        using BaseType = DnnlLayer<data_t>;
        using DnnlMemory = typename BaseType::DnnlMemory;

        using BaseType::_input;
        using BaseType::_inputGradient;

        using BaseType::_output;
        using BaseType::_outputGradient;

        using BaseType::_forwardStream;
        using BaseType::_backwardStream;

        using BaseType::_engine;

        using BaseType::_typeTag;

        dnnl::memory::dims _poolingStride;
        dnnl::memory::dims _poolingWindow;
        dnnl::memory::dims _poolingPadding;
        DnnlMemory _workspaceMemory;

        dnnl::pooling_forward::primitive_desc _forwardPrimitiveDescriptor;
        dnnl::pooling_backward::primitive_desc _backwardPrimitiveDescriptor;
    };
} // namespace elsa