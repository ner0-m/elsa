#include "DnnlLayer.h"

namespace elsa
{
    template <typename data_t>
    DnnlLayer<data_t>::DnnlLayer(const DataDescriptor& inputDescriptor,
                                 const DataDescriptor& outputDescriptor)
        : _outputDescriptor(outputDescriptor.clone()),
          _inputDescriptor(inputDescriptor.clone()),
          _engine(std::make_shared<dnnl::engine>(dnnl::engine::kind::cpu, 0))
    {
        // Set source memory descriptor
        for (const auto& dim : inputDescriptor.getNumberOfCoefficientsPerDimension())
            _input.dimensions.push_back(dim);

        _inputGradient.dimensions = _input.dimensions;

        _input.formatTag = dataDescriptorToDnnlMemoryFormatTag(inputDescriptor, true);
        _inputGradient.formatTag = _input.formatTag;

        if (_input.formatTag == dnnl::memory::format_tag::undef)
            throw std::logic_error("Could not resolve Dnnl source memory format tag");

        _input.descriptor =
            dnnl::memory::desc({_input.dimensions}, _typeTag, dnnl::memory::format_tag::any);

        _inputGradient.descriptor = _input.descriptor;

        // Set destination memory descriptor
        for (const auto& dim : outputDescriptor.getNumberOfCoefficientsPerDimension())
            _output.dimensions.push_back(dim);

        _outputGradient.dimensions = _output.dimensions;

        _output.descriptor =
            dnnl::memory::desc({_output.dimensions}, _typeTag, dnnl::memory::format_tag::any);
        _outputGradient.descriptor = _output.descriptor;

        _output.formatTag = dataDescriptorToDnnlMemoryFormatTag(outputDescriptor, true);
        _outputGradient.formatTag = _output.formatTag;
    }

    template <typename data_t>
    void DnnlLayer<data_t>::writeToDnnlMemory(const DataContainer<data_t>& data,
                                              dnnl::memory& memory)
    {
        data_t* dst = static_cast<data_t*>(memory.get_data_handle());
        for (int i = 0; i < data.getDataDescriptor().getNumberOfCoefficients(); ++i)
            dst[i] = data[i];
    }

    template <typename data_t>
    void DnnlLayer<data_t>::readFromDnnlMemory(DataContainer<data_t>& data,
                                               const dnnl::memory& memory)
    {
        data_t* src = static_cast<data_t*>(memory.get_data_handle());
        for (int i = 0; i < data.getDataDescriptor().getNumberOfCoefficients(); ++i)
            data[i] = src[i];
    }

    template <typename data_t>
    dnnl::memory::format_tag
        DnnlLayer<data_t>::dataDescriptorToDnnlMemoryFormatTag(const DataDescriptor& desc,
                                                               bool isInput)
    {
        using ft = dnnl::memory::format_tag;

        switch (desc.getNumberOfDimensions()) {
            case 2:
                return (isInput ? ft::nc : ft::oi);
            case 3:
                return (isInput ? ft::ncw : ft::oiw);
            case 4:
                return (isInput ? ft::nchw : ft::oihw);
            case 5:
                return (isInput ? ft::ncdhw : ft::oidhw);
            default:
                return ft::undef;
        }
    }

    template <typename data_t>
    void DnnlLayer<data_t>::forwardPropagate(dnnl::stream& executionStream)
    {
        if (_forwardStream.primitives.size() != _forwardStream.arguments.size())
            throw std::logic_error(
                "Number of Dnnl primitives and number of primitive arguments must match");

        for (std::size_t i = 0; i < _forwardStream.primitives.size(); ++i)
            _forwardStream.primitives[i].execute(executionStream, _forwardStream.arguments[i]);
    }

    template <typename data_t>
    void DnnlLayer<data_t>::backwardPropagate(dnnl::stream& executionStream)
    {
        if (_backwardStream.primitives.size() != _backwardStream.arguments.size())
            throw std::logic_error(
                "Number of Dnnl primitives and number of primitive arguments must match");

        for (std::size_t i = 0; i < _backwardStream.primitives.size(); ++i)
            _backwardStream.primitives[i].execute(executionStream, _backwardStream.arguments[i]);
    } // namespace elsa

    template <typename data_t>
    std::shared_ptr<dnnl::engine> DnnlLayer<data_t>::getEngine() const
    {
        return _engine;
    }

    template <typename data_t>
    void DnnlLayer<data_t>::setEngine(std::shared_ptr<dnnl::engine> engine)
    {
        _engine = engine;
    }

    template <typename data_t>
    void DnnlLayer<data_t>::setInput(const DataContainer<data_t>& input)
    {
        // If no input has been set yet we allocate
        if (!_input.describedMemory) {
            _input.describedMemory = std::make_shared<dnnl::memory>(
                dnnl::memory::desc({{_input.dimensions}, _typeTag, _input.formatTag}), *_engine);
        }

        writeToDnnlMemory(input, *_input.describedMemory);
    }

    template <typename data_t>
    void DnnlLayer<data_t>::setSourceMemory(std::shared_ptr<dnnl::memory> input)
    {
        // Set source memory
        _input.describedMemory = input;
    }

    template <typename data_t>
    DataContainer<data_t> DnnlLayer<data_t>::getOutput() const
    {
        DataContainer<data_t> output(*_outputDescriptor);

        // TODO: Check if we really need this reorder based on forwardPrimitve.dst_desc(). This can
        // potentially safe a copy.

        // If memory has been reordered, we have to check whether output
        // memory needs to be also reordered
        auto outMem = *_output.effectiveMemory;
        if (_output.wasReordered) {
            // We reorder directly and open a new execution stream for this. Note that this could be
            // relatively expensive and should be used for reporting the final net output or for
            // debugging purposes only
            outMem = dnnl::memory({{_output.dimensions}, _typeTag, _output.formatTag}, *_engine);
            dnnl::stream execStream(*_engine);
            dnnl::reorder(*_output.effectiveMemory, outMem)
                .execute(execStream,
                         {{DNNL_ARG_FROM, *_output.effectiveMemory}, {DNNL_ARG_TO, outMem}});
            execStream.wait();
        }

        // Write reordered memory to output DataContainer. This performs a copy.
        readFromDnnlMemory(output, outMem);
        return output;
    }

    template <typename data_t>
    void DnnlLayer<data_t>::compile(PropagationKind propagation)
    {
        if (!_engine)
            throw std::logic_error("Failed to compile layer: Dnnl engine is null");
        if (!_input.describedMemory)
            throw std::logic_error("Failed to compile layer: Dnnl source memory is null");

        // If this layer may not reorder source or destination memory, we equal
        // the pointers of described end effective memory
        if (!_input.canBeReordered) {
            _input.effectiveMemory = _input.describedMemory;
            _input.descriptor = _input.describedMemory->get_desc();
        }

        switch (propagation) {
            case PropagationKind::Forward:
                if (!_forwardStream.isCompiled)
                    compileForwardStream();
                break;
            case PropagationKind::Backward:
                if (!_backwardStream.isCompiled)
                    compileBackwardStream();
                break;
            case PropagationKind::Full:
                if (!_forwardStream.isCompiled)
                    compileForwardStream();
                if (!_backwardStream.isCompiled)
                    compileBackwardStream();
                break;
            default:
                throw std::invalid_argument("Failed to compile layer: Unkown propagation kind");
        }
    }

    template <typename data_t>
    std::shared_ptr<dnnl::memory> DnnlLayer<data_t>::getOutputMemory()
    {
        return _output.effectiveMemory;
    }

    template <typename data_t>
    void DnnlLayer<data_t>::setOutputGradient(const DataContainer<data_t>& gradient)
    {
        _outputGradient.describedMemory = std::make_shared<dnnl::memory>(
            dnnl::memory::desc({{_outputGradient.dimensions}, _typeTag, _outputGradient.formatTag}),
            *_engine);
        writeToDnnlMemory(gradient, *_outputGradient.describedMemory);
    }

    template <typename data_t>
    DataContainer<data_t> DnnlLayer<data_t>::getInputGradient() const
    {
        if (!_inputGradient.effectiveMemory)
            throw std::logic_error("Cannot get input gradient because it is empty");

        DataContainer<data_t> output(*_inputDescriptor);
        dnnl::memory outMem;
        if (_inputGradient.effectiveMemory->get_desc() != _inputGradient.descriptor) {
            outMem = dnnl::memory({{_inputGradient.dimensions}, _typeTag, _inputGradient.formatTag},
                                  *_engine);
            dnnl::stream execStream(*_engine);
            dnnl::reorder(*_inputGradient.effectiveMemory, outMem)
                .execute(execStream,
                         {{DNNL_ARG_FROM, *_inputGradient.effectiveMemory}, {DNNL_ARG_TO, outMem}});
            execStream.wait();
        }

        // Write reordered memory to output DataContainer. This performs a copy.
        readFromDnnlMemory(output, outMem);
        return output;
    }

    template <typename data_t>
    void DnnlLayer<data_t>::reorderMemory(const dnnl::memory::desc& memoryDesc,
                                          DnnlLayer<data_t>::DnnlMemory& memory,
                                          DnnlLayer<data_t>::PropagationStream& stream)
    {
        memory.effectiveMemory = memory.describedMemory;
        if (memory.describedMemory->get_desc() != memoryDesc) {
            memory.wasReordered = true;
            memory.effectiveMemory = std::make_shared<dnnl::memory>(memoryDesc, *_engine);
            stream.primitives.push_back(
                dnnl::reorder(*memory.describedMemory, *memory.effectiveMemory));
            stream.arguments.push_back(
                {{DNNL_ARG_FROM, *memory.describedMemory}, {DNNL_ARG_TO, *memory.effectiveMemory}});
        }
    }

    template class DnnlLayer<float>;
} // namespace elsa
