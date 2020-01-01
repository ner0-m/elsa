#include "DnnlLayer.h"
#include <iostream>

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
        : _engine(std::make_shared<dnnl::engine>(dnnl::engine::kind::cpu, 0))
    {
        // Set source memory descriptor
        for (const auto& dim : inputDescriptor.getNumberOfCoefficientsPerDimension())
            _srcMemoryDimensions.push_back(dim);

        _srcMemoryFormatTag = dataDescriptorToDnnlMemoryFormatTag(inputDescriptor, true);

        _srcMemoryDescriptor =
            dnnl::memory::desc({_srcMemoryDimensions}, _typeTag, dnnl::memory::format_tag::any);

        // Set destination memory descriptor
        for (const auto& dim : outputDescriptor.getNumberOfCoefficientsPerDimension())
            _dstMemoryDimensions.push_back(dim);

        _dstMemoryDescriptor =
            dnnl::memory::desc({_dstMemoryDimensions}, _typeTag, dnnl::memory::format_tag::any);
    }

    template <typename data_t>
    void DnnlLayer<data_t>::forwardPropagate(dnnl::stream& executionStream)
    {
        if (_forwardPrimitives.size() != _forwardArguments.size())
            throw std::logic_error("Number of Dnnl primitives and number of arguments must match");

        for (int i = 0; i < _forwardPrimitives.size(); ++i)
            _forwardPrimitives[i].execute(executionStream, _forwardArguments[i]);
    }

    template <typename data_t>
    std::shared_ptr<dnnl::engine> DnnlLayer<data_t>::getEngine() const
    {
        return _engine;
    }

    template <typename data_t>
    void DnnlLayer<data_t>::setInput(const DataContainer<data_t>& input)
    {
        _srcMemory = std::make_shared<dnnl::memory>(
            dnnl::memory::desc({{_srcMemoryDimensions}, _typeTag, _srcMemoryFormatTag}), *_engine);
        writeToDnnlMemory(input, *_srcMemory);
    }

    template <typename data_t>
    void DnnlLayer<data_t>::setInput(const dnnl::memory& input)
    {
        if (input.get_desc() != _srcMemoryDescriptor)
            throw std::invalid_argument("Descriptor of user input memory doesn't match descriptor "
                                        "of layer's source memory");
        _srcMemory = std::make_shared<dnnl::memory>(input);
    }

    template <typename data_t>
    DataContainer<data_t> DnnlLayer<data_t>::getOutput()
    {
        IndexVector_t outVec(_dstMemoryDimensions.size());
        for (int i = 0; i < outVec.size(); ++i)
            outVec[i] = _dstMemoryDimensions[i];
        DataDescriptor outDesc(outVec);
        DataContainer<data_t> output(outDesc);

        // If memory has been reordered, we have to check whether output memory
        // needs to be also reordered
        auto outMem = _dstMemory;
        if (_hasReorderedMemory) {
            // We reorder directly. Note that this could be relatively expensive and should be used
            // for reporting the final net output or for debugging purposes only
            outMem =
                dnnl::memory({{_dstMemoryDimensions}, _typeTag, _srcMemoryFormatTag}, *_engine);
            dnnl::stream s(*_engine);
            dnnl::reorder(_dstMemory, outMem)
                .execute(s, {{DNNL_ARG_FROM, _dstMemory}, {DNNL_ARG_TO, outMem}});
        }
        readFromDnnlMemory(output, outMem);
        return output;
    }

    template <typename data_t>
    void DnnlLayer<data_t>::compile()
    {
        if (!_srcMemory)
            throw std::logic_error("Dnnl source memory must not be null");
        if (!_engine)
            throw std::logic_error("Dnnl engine must not be null");
    }

    template class DnnlLayer<float>;
} // namespace elsa
