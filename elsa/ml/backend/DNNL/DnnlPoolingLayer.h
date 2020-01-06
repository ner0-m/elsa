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

        using BaseType = DnnlLayer<data_t>;

        using BaseType::_engine;
        using BaseType::_srcMemoryDescriptor;
        using BaseType::_dstMemoryDescriptor;
        using BaseType::_dstMemory;
        using BaseType::_srcMemory;
        using BaseType::_forwardArguments;
        using BaseType::_forwardPrimitives;

        dnnl::pooling_forward::primitive_desc _forwardPrimitiveDescriptor;
        dnnl::memory::dims _poolingStride;
        dnnl::memory::dims _poolingWindow;
        dnnl::memory::dims _poolingPadding;
        dnnl::memory _workspaceMemory;
    };
} // namespace elsa