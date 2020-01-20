#pragma once

#include "DnnlLayer.h"
#include "elsaDefines.h"
#include "JosephsMethod.h"
#include "SiddonsMethod.h"
#include "LinearOperator.h"
#include "DataContainer.h"

namespace elsa
{
    template <typename data_t>
    class DnnlFixedLayer final : public DnnlLayer<data_t>
    {
    public:
        using BaseType = DnnlLayer<data_t>;

        DnnlFixedLayer(const DataDescriptor& inputDescriptor,
                       const DataDescriptor& outputDescriptor, const JosephsMethod<data_t>& op);

        void forwardPropagate(dnnl::stream& executionStream) override;

        void backwardPropagate(dnnl::stream& executionStream) override;

        void initialize() override;

    private:
        bool _isInitialized = false;

        /// DnnlLayer::compileForwardStream
        void compileForwardStream() override;

        /// DnnlLayer::compileBackwardStream
        void compileBackwardStream() override;

        /// DnnlLayer::_input
        using BaseType::_input;

        /// DnnlLayer::_inputGradient
        using BaseType::_inputGradient;

        /// DnnlLayer::_output
        using BaseType::_output;

        /// DnnlLayer::_outputGradient
        using BaseType::_outputGradient;

        /// \copydoc DnnlLayer::_engine;
        using BaseType::_engine;

        /// \copydoc DnnlLayer::_typeTag;
        using BaseType::_typeTag;

        /// \copydoc DnnlLayer::_inputDescriptor;
        using BaseType::_inputDescriptor;

        /// \copydoc DnnlLayer::_outputDescriptor;
        using BaseType::_outputDescriptor;

        std::unique_ptr<DataDescriptor> _operatorInputDescriptor;
        std::unique_ptr<DataDescriptor> _operatorOutputDescriptor;

        std::unique_ptr<LinearOperator<data_t>> _operator;
    };
} // namespace elsa
