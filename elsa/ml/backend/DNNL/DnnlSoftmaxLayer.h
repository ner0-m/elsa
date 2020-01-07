#pragma once

#include "DnnlLayer.h"

namespace elsa
{
    template <typename data_t>
    class DnnlSoftmaxLayer : public DnnlLayer<data_t>
    {
    public:
        DnnlSoftmaxLayer(const DataDescriptor& inputDescriptor,
                         const DataDescriptor& outputDescriptor);

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

        int _softmaxAxis;

        dnnl::softmax_forward::primitive_desc _forwardPrimitiveDescriptor;
    };
} // namespace elsa