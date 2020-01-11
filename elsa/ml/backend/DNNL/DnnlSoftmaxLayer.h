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

        using BaseType::_input;
        using BaseType::_inputGradient;

        using BaseType::_output;
        using BaseType::_outputGradient;

        using BaseType::_forwardStream;
        using BaseType::_backwardStream;

        using BaseType::_engine;

        int _softmaxAxis;

        dnnl::softmax_forward::primitive_desc _forwardPrimitiveDescriptor;
    };
} // namespace elsa