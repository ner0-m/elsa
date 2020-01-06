#include "DnnlLayer.h"

namespace elsa
{
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
    DnnlLayer<data_t>::DnnlLayer(const DataDescriptor& inputDescriptor,
                                 const DataDescriptor& outputDescriptor)
        : _engine(std::make_shared<dnnl::engine>(dnnl::engine::kind::cpu, 0)),
          _outputDescriptor(outputDescriptor.clone())
    {
        // Set source memory descriptor
        for (const auto& dim : inputDescriptor.getNumberOfCoefficientsPerDimension())
            _srcMemoryDimensions.push_back(dim);

        _srcMemoryFormatTag = dataDescriptorToDnnlMemoryFormatTag(inputDescriptor, true);

        if (_srcMemoryFormatTag == dnnl::memory::format_tag::undef)
            throw std::logic_error("Could not resolve Dnnl source memory format tag");

        _srcMemoryDescriptor =
            dnnl::memory::desc({_srcMemoryDimensions}, _typeTag, dnnl::memory::format_tag::any);

        // Set destination memory descriptor
        for (const auto& dim : outputDescriptor.getNumberOfCoefficientsPerDimension())
            _dstMemoryDimensions.push_back(dim);

        _dstMemoryDescriptor =
            dnnl::memory::desc({_dstMemoryDimensions}, _typeTag, dnnl::memory::format_tag::any);

        // // Set diff src memory descriptor
        // _diffSrcMemoryDescriptor = _srcMemoryDescriptor;

        // // Set diff dst memory descriptor
        // _diffDstMemoryDescriptor = _dstMemoryDescriptor;
    }

    template <typename data_t>
    void DnnlLayer<data_t>::forwardPropagate(dnnl::stream& executionStream)
    {
        if (_forwardPrimitives.size() != _forwardArguments.size())
            throw std::logic_error(
                "Number of Dnnl primitives and number of primitive arguments must match");

        for (std::size_t i = 0; i < _forwardPrimitives.size(); ++i) {
            _forwardPrimitives[i].execute(executionStream, _forwardArguments[i]);
        }
    }

    template <typename data_t>
    void DnnlLayer<data_t>::backwardPropagate(dnnl::stream& executionStream)
    {
        if (_backwardPrimitives.size() != _backwardArguments.size())
            throw std::logic_error(
                "Number of Dnnl primitives and number of primitive arguments must match");

        for (std::size_t i = 0; i < _backwardPrimitives.size(); ++i) {
            _backwardPrimitives[i].execute(executionStream, _backwardArguments[i]);
        }
    }

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
        _srcMemory = std::make_shared<dnnl::memory>(
            dnnl::memory::desc({{_srcMemoryDimensions}, _typeTag, _srcMemoryFormatTag}), *_engine);
        writeToDnnlMemory(input, *_srcMemory);
    }

    template <typename data_t>
    void DnnlLayer<data_t>::setSourceMemory(std::shared_ptr<dnnl::memory> input)
    {
        // Set source memory
        _srcMemory = input;
    }

    template <typename data_t>
    DataContainer<data_t> DnnlLayer<data_t>::getOutput() const
    {
        DataContainer<data_t> output(*_outputDescriptor);

        // If memory has been reordered, we have to check whether output memory
        // needs to be also reordered
        auto outMem = *_dstMemory;
        if (_hasReorderedMemory) {
            // We reorder directly and open a new execution stream for this. Note that this could be
            // relatively expensive and should be used for reporting the final net output or for
            // debugging purposes only
            outMem = dnnl::memory({{_dstMemoryDimensions},
                                   _typeTag,
                                   dataDescriptorToDnnlMemoryFormatTag(*_outputDescriptor, true)},
                                  *_engine);
            dnnl::stream execStream(*_engine);
            dnnl::reorder(*_dstMemory, outMem)
                .execute(execStream, {{DNNL_ARG_FROM, *_dstMemory}, {DNNL_ARG_TO, outMem}});
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
        if (!_srcMemory)
            throw std::logic_error("Failed to compile layer: Dnnl source memory is null");

        switch (propagation) {
            case PropagationKind::Forward:
                compileForwardStream();
                break;
            case PropagationKind::Backward:
                compileBackwardStream();
                break;
            case PropagationKind::Full:
                compileForwardStream();
                compileBackwardStream();
                break;
            default:
                throw std::invalid_argument("Failed to compile layer: Unkown propagation kind");
        }
    }

    template <typename data_t>
    std::shared_ptr<dnnl::memory> DnnlLayer<data_t>::getOutputMemory()
    {
        return _dstMemory;
    }

    template class DnnlLayer<float>;
} // namespace elsa
