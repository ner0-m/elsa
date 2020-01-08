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
        void compileForwardStream() override;

        using BaseType = DnnlLayer<data_t>;

        using BaseType::_engine;
        using BaseType::_srcMemoryDescriptor;
        using BaseType::_dstMemoryDescriptor;
        using BaseType::_dstMemory;
        using BaseType::_srcMemory;
        using BaseType::_forwardArguments;
        using BaseType::_forwardPrimitives;

        dnnl::memory _workspaceMemory;

        dnnl::memory::dim _localSize;

        data_t _alpha;
        data_t _beta;
        data_t _k;

        dnnl::lrn_forward::primitive_desc _forwardPrimitiveDescriptor;
    };
} // namespace elsa