#include "DnnlLayer.h"
#include "TypeCasts.hpp"
#include <iostream>
#include <sstream>

namespace elsa::ml
{
    namespace detail
    {
        template <typename data_t>
        DnnlLayer<data_t>::DnnlLayer(const VolumeDescriptor& inputDescriptor,
                                     const VolumeDescriptor& outputDescriptor,
                                     const std::string& name, int allowedNumberOfInputs)
            : DnnlLayer(std::vector<VolumeDescriptor>{inputDescriptor}, outputDescriptor, name,
                        allowedNumberOfInputs)
        {
        }

        template <typename data_t>
        DnnlLayer<data_t>::DnnlLayer(const std::vector<VolumeDescriptor>& inputDescriptor,
                                     const VolumeDescriptor& outputDescriptor,
                                     const std::string& name, int allowedNumberOfInputs)
            : _input(inputDescriptor.size()),
              _inputGradient(inputDescriptor.size()),
              _outputGradient(1), // we need at least one output-gradient
              _outputDescriptor(outputDescriptor.clone()),
              _engine(std::make_shared<dnnl::engine>(dnnl::engine::kind::cpu, 0)),
              _allowedNumberOfInputs(allowedNumberOfInputs),
              _name(name)
        {
            // A layer can have several inputs but only one single output.
            // However, for the gradients the situation is different: A layer
            // can have multiple output-gradients (gradients coming from different
            // connected layers after the current layer) and multiple input-gradients
            // (e.g. in the case of a concatenation-layer).

            // Set input descriptors and input dimensions
            for (std::size_t i = 0; i < inputDescriptor.size(); ++i) {
                // Clone input-descriptor
                _inputDescriptor.push_back(inputDescriptor[i].clone());

                // Get memory-format tag for input and input-gradient
                _input[asUnsigned(i)].formatTag =
                    dataDescriptorToDnnlMemoryFormatTag(inputDescriptor[asUnsigned(i)], true);
                _inputGradient[asUnsigned(i)].formatTag = _input[asUnsigned(i)].formatTag;

                assert(_input[asUnsigned(i)].formatTag != dnnl::memory::format_tag::undef
                       && "Could not resolve Dnnl source memory format tag");

                // Get input and input-gradient dimensions
                for (const auto& dim :
                     _inputDescriptor[asUnsigned(i)]->getNumberOfCoefficientsPerDimension()) {
                    _input[asUnsigned(i)].dimensions.push_back(dim);
                    _inputGradient[asUnsigned(i)].dimensions.push_back(dim);
                }

                // Get input and input-gradient Dnnl descriptors
                _input[asUnsigned(i)].descriptor = dnnl::memory::desc(
                    {_input[asUnsigned(i)].dimensions}, _typeTag, dnnl::memory::format_tag::any);

                _inputGradient[asUnsigned(i)].descriptor = _input[asUnsigned(i)].descriptor;
            }

            // Set output memory descriptor
            for (const auto& dim : outputDescriptor.getNumberOfCoefficientsPerDimension())
                _output.dimensions.push_back(dim);

            // The shape of all output-gradients match the shape of the single
            // layer output, only the memory can be different
            _output.descriptor =
                dnnl::memory::desc({_output.dimensions}, _typeTag, dnnl::memory::format_tag::any);
            _output.formatTag = dataDescriptorToDnnlMemoryFormatTag(outputDescriptor, true);

            for (auto&& outGrad : _outputGradient) {
                outGrad.dimensions = _output.dimensions;
                outGrad.descriptor = _output.descriptor;
                outGrad.formatTag = _output.formatTag;
            }
        }

        template <typename data_t>
        void DnnlLayer<data_t>::writeToDnnlMemory(const DataContainer<data_t>& data,
                                                  dnnl::memory& memory)
        {
            assert(data.getSize() == memory.get_desc().get_size() / sizeof(data_t));
            assert(memory.get_data_handle() != nullptr);

            data_t* dst = static_cast<data_t*>(memory.get_data_handle());
            for (int i = 0; i < data.getSize(); ++i)
                dst[i] = data[i];
        }

        template <typename data_t>
        void DnnlLayer<data_t>::readFromDnnlMemory(DataContainer<data_t>& data,
                                                   const dnnl::memory& memory)
        {
            assert(data.getSize() == memory.get_desc().get_size() / sizeof(data_t));
            assert(memory.get_data_handle() != nullptr);
            const data_t* src = static_cast<const data_t*>(memory.get_data_handle());
            for (int i = 0; i < data.getSize(); ++i)
                data[i] = src[i];
        }

        template <typename data_t>
        dnnl::memory::format_tag
            DnnlLayer<data_t>::dataDescriptorToDnnlMemoryFormatTag(const VolumeDescriptor& desc,
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
        std::string DnnlLayer<data_t>::dnnlMemoryFormatTagToString(dnnl::memory::format_tag tag)
        {

            auto formatStr = [](const std::string& input, const std::string& weights) {
                std::stringstream ss;
                ss << "dnnl::memory:format_tag::" << input
                   << " (input), dnnl::memory:format_tag::" << weights << " (weights)";
                return ss.str();
            };

            using ft = dnnl::memory::format_tag;

            switch (tag) {
                case ft::undef:
                    return formatStr("undef", "undef");
                case ft::nc:
                    return formatStr("nc", "oi");
                case ft::ncw:
                    return formatStr("ncw", "oiw");
                case ft::nchw:
                    return formatStr("nchw", "oihw");
                case ft::ncdhw:
                    return formatStr("ncdhw", "oidhw");
                default:
                    assert(false && "This execution path of the code should never be reached");
            }
            assert(false && "This execution path of the code should never be reached");
            return "";
        }

        template <typename data_t>
        void DnnlLayer<data_t>::forwardPropagate(dnnl::stream& executionStream)
        {
            Logger::get(_name)->trace("Forward propagate");
            assert(_input.size() == _allowedNumberOfInputs
                   || _allowedNumberOfInputs == DnnlLayer::anyNumberOfInputs
                          && "Too many inputs provided");

            assert(_input.size() == _inputDescriptor.size()
                   && "Number of provided inputs does not match number of input-descriptors");

            assert(_forwardStream.isCompiled
                   && "Cannot forward propagate because forward-stream has not been compiled");

            assert(_forwardStream.primitives.size() == _forwardStream.arguments.size()
                   && "Number of Dnnl primitives and number of primitive arguments must match");

            for (std::size_t i = 0; i < _forwardStream.primitives.size(); ++i)
                _forwardStream.primitives[i].execute(executionStream, _forwardStream.arguments[i]);

            if (needsForwardSynchronisation()) {
                executionStream.wait();
            }
        }

        template <typename data_t>
        void DnnlLayer<data_t>::backwardPropagate(dnnl::stream& executionStream)
        {
            Logger::get(_name)->trace("Backward propagate");
            assert(_input.size() == _allowedNumberOfInputs
                   || _allowedNumberOfInputs == DnnlLayer::anyNumberOfInputs
                          && "Too many inputs provided");

            assert(_backwardStream.isCompiled
                   && "Cannot backward propagate because backward-stream has not been compiled");

            assert(_backwardStream.primitives.size() == _backwardStream.arguments.size()
                   && "Number of Dnnl primitives and number of primitive arguments must match");

            for (std::size_t i = 0; i < _backwardStream.primitives.size(); ++i)
                _backwardStream.primitives[i].execute(executionStream,
                                                      _backwardStream.arguments[i]);

            if (needsBackwardSynchronisation()) {
                executionStream.wait();
            }
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
        void DnnlLayer<data_t>::setInput(const DataContainer<data_t>& input, index_t index)
        {
            Logger::get(_name)->trace("Set layer input from DataContainer at index {}", index);
            // Check if index is valid
            validateVectorIndex(_input, index);

            // If no input has been set yet we allocate
            if (!_input[index].describedMemory) {
                _input[index].describedMemory = std::make_shared<dnnl::memory>(
                    dnnl::memory::desc(
                        {{_input[index].dimensions}, _typeTag, _input[index].formatTag}),
                    *_engine);
            }

            writeToDnnlMemory(input, *_input[index].describedMemory);
        }

        template <typename data_t>
        void DnnlLayer<data_t>::setInputMemory(std::shared_ptr<dnnl::memory> input, index_t index)
        {
            Logger::get(_name)->trace("Set layer input memory at index {}", index);

            // Check if index is valid
            validateVectorIndex(_input, index);

            // Set input memory
            _input[index].describedMemory = input;
            validateDnnlMemory(_input[index].describedMemory);
        }

        template <typename data_t>
        void DnnlLayer<data_t>::setNextInputMemory(std::shared_ptr<dnnl::memory> input)
        {
            index_t nextIndex = _currentInputMemoryIndex++;
            setInputMemory(input, nextIndex);
        }

        template <typename data_t>
        void DnnlLayer<data_t>::setOutputGradient(const DataContainer<data_t>& gradient,
                                                  index_t index)
        {
            // Check if index is valid
            validateVectorIndex(_outputGradient, index);

            if (!_outputGradient[index].describedMemory) {
                _outputGradient[index].describedMemory = std::make_shared<dnnl::memory>(
                    dnnl::memory::desc({{_outputGradient[index].dimensions},
                                        _typeTag,
                                        _outputGradient[index].formatTag}),
                    *_engine);
            }
            writeToDnnlMemory(gradient, *_outputGradient[index].describedMemory);
        }

        template <typename data_t>
        void
            DnnlLayer<data_t>::setOutputGradientMemory(std::shared_ptr<dnnl::memory> outputGradient,
                                                       index_t index)
        {
            // Check if index is valid
            validateVectorIndex(_outputGradient, index);

            // Set output-gradient memory
            _outputGradient[index].describedMemory = outputGradient;
            validateDnnlMemory(_outputGradient[index].describedMemory);
        }

        template <typename data_t>
        void DnnlLayer<data_t>::setNextOutputGradientMemory(
            std::shared_ptr<dnnl::memory> outputGradient)
        {
            index_t nextIndex = _currentOutputGradientMemoryIndex++;
            setOutputGradientMemory(outputGradient, nextIndex);
        }

        /// Reverse a volume-descriptor
        ///
        /// If we have a descriptor
        ///   {w, h, c, n}
        /// this creates a descriptor
        ///   {n, c, h, w}.
        static inline VolumeDescriptor reverseDataDescriptor(const DataDescriptor& desc)
        {
            IndexVector_t dims = desc.getNumberOfCoefficientsPerDimension().reverse();
            return VolumeDescriptor(dims);
        }

        template <typename data_t>
        DataContainer<data_t> DnnlLayer<data_t>::getOutput() const
        {
            DataContainer<data_t> output(reverseDataDescriptor(*_outputDescriptor));

            // TODO(tellenbach): Check if we really need this reorder based on
            // forwardPrimitve.dst_desc(). This can potentially safe a copy.

            // If memory has been reordered, we have to check whether output
            // memory needs to be also reordered
            // TODO(tellenbach): Add reordering to layer compilation
            auto outMem = *getOutputMemory();
            if (_output.wasReordered) {
                // We reorder directly and open a new execution stream for this. Note that this
                // could be relatively expensive and should be used for reporting the final net
                // output or for traceging purposes only
                outMem =
                    dnnl::memory({{_output.dimensions}, _typeTag, _output.formatTag}, *_engine);
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
        void DnnlLayer<data_t>::compileForwardStream()
        {
            for (auto&& input : _input) {
                if (!input.describedMemory) {
                    input.describedMemory = std::make_shared<dnnl::memory>(
                        dnnl::memory::desc({{input.dimensions}, _typeTag, input.formatTag}),
                        *_engine);
                }

                // If this layer may not reorder source or destination memory, we equal
                // the pointers of described end effective memory
                if (!input.canBeReordered) {
                    input.effectiveMemory = input.describedMemory;
                    input.descriptor = input.describedMemory->get_desc();
                }
            }
            _forwardStream.isCompiled = true;
        }

        template <typename data_t>
        void DnnlLayer<data_t>::compileBackwardStream()
        {
            Logger::get(_name)->trace("Compile backward stream (base)");

            for (auto&& outGrad : _outputGradient) {
                if (!outGrad.describedMemory) {
                    outGrad.describedMemory = std::make_shared<dnnl::memory>(
                        dnnl::memory::desc({{outGrad.dimensions}, _typeTag, outGrad.formatTag}),
                        *_engine);
                }

                // If this layer may not reorder source or destination memory, we equal
                // the pointers of described end effective memory
                if (!outGrad.canBeReordered) {
                    outGrad.effectiveMemory = outGrad.describedMemory;
                    outGrad.descriptor = outGrad.describedMemory->get_desc();
                }
            }

            // Handle multiple output-gradients
            handleMultipleOutputGradients();

            assert(_outputGradient.size() != 0
                   && "Cannot compile backward-stream without output gradient");
            _backwardStream.isCompiled = true;
        }

        template <typename data_t>
        void DnnlLayer<data_t>::compile(PropagationKind propagation)
        {
            assert(_engine != nullptr && "Failed to compile layer: Dnnl engine is null");

            switch (propagation) {
                case PropagationKind::Forward:
                    if (!_forwardStream.isCompiled)
                        compileForwardStream();
                    break;
                case PropagationKind::Backward:
                case PropagationKind::Full:
                    if (!_forwardStream.isCompiled)
                        compileForwardStream();
                    if (!_backwardStream.isCompiled) {
                        compileBackwardStream();
                    }
                    break;
                default:
                    assert(false && "This execution path of the code should never be reached");
            }
        }

        template <typename data_t>
        std::shared_ptr<dnnl::memory> DnnlLayer<data_t>::getOutputMemory() const
        {
            validateDnnlMemory(_output.effectiveMemory);
            return _output.effectiveMemory;
        }

        template <typename data_t>
        std::shared_ptr<dnnl::memory> DnnlLayer<data_t>::getInputGradientMemory(index_t index)
        {
            validateVectorIndex(_inputGradient, index);
            validateDnnlMemory(_inputGradient[asUnsigned(index)].effectiveMemory);
            return _inputGradient[asUnsigned(index)].effectiveMemory;
        }

        template <typename data_t>
        DataContainer<data_t> DnnlLayer<data_t>::getInputGradient(index_t index) const
        {
            validateVectorIndex(_inputGradient, index);
            validateDnnlMemory(_inputGradient[index].effectiveMemory);

            DataContainer<data_t> output(reverseDataDescriptor(*_inputDescriptor[index]));

            dnnl::memory outMem;
            if (_inputGradient[index].effectiveMemory->get_desc()
                != _inputGradient[index].descriptor) {
                outMem = dnnl::memory(
                    {{_inputGradient[index].dimensions}, _typeTag, _inputGradient[index].formatTag},
                    *_engine);
                dnnl::stream execStream(*_engine);
                dnnl::reorder(*_inputGradient[index].effectiveMemory, outMem)
                    .execute(execStream, {{DNNL_ARG_FROM, *_inputGradient[index].effectiveMemory},
                                          {DNNL_ARG_TO, outMem}});
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
            validateDnnlMemory(memory.describedMemory);
            // Default case: effective memory and described memory are the same
            memory.effectiveMemory = memory.describedMemory;

            // We need reordering if the memory description differs from the description defined by
            // the primitive. In this case we reorder from the manual description to the one chosen
            // by Dnnl (via a primitive)
            if (memory.describedMemory->get_desc() != memoryDesc) {
                memory.wasReordered = true;
                memory.effectiveMemory = std::make_shared<dnnl::memory>(memoryDesc, *_engine);

                // Add reordering primitive and its arguments to the execution stream
                ELSA_ML_ADD_DNNL_PRIMITIVE(
                    stream, dnnl::reorder(*memory.describedMemory, *memory.effectiveMemory));
                stream.arguments.push_back({{DNNL_ARG_FROM, *memory.describedMemory},
                                            {DNNL_ARG_TO, *memory.effectiveMemory}});
            }
        }

        template <typename data_t>
        bool DnnlLayer<data_t>::isTrainable() const
        {
            return false;
        }

        template <typename data_t>
        bool DnnlLayer<data_t>::canMerge() const
        {
            return false;
        }

        template <typename data_t>
        void DnnlLayer<data_t>::handleMultipleOutputGradients()
        {
            // Check that all output-gradient descriptors are equal and that
            // they match this layer's output-descriptor
            assert(!_outputGradient.empty() && "List of output-gradients is empty");
            assert(std::adjacent_find(
                       _outputGradient.begin(), _outputGradient.end(),
                       [](const auto& a, const auto& b) { return a.dimensions != b.dimensions; })
                       == _outputGradient.end()
                   && "All output-gradient descriptors must be equal");
            assert(_outputGradient.front().dimensions == _output.dimensions
                   && "Dimensions of output-gradients must match dimensions of output");

            if (_outputGradient.size() > 1) {
                Logger::get(_name)->trace("Found multiple output-gradients");
                std::vector<dnnl::memory> mem;
                std::vector<dnnl::memory::desc> memDesc;
                for (std::size_t i = 0; i < _outputGradient.size(); ++i) {
                    memDesc.push_back(_outputGradient[i].descriptor);
                    validateDnnlMemory(_outputGradient[i].effectiveMemory);
                    mem.push_back(*_outputGradient[i].effectiveMemory);
                }

                // Do not scale during summation
                std::vector<data_t> scales(_outputGradient.size(), data_t(1));

                // Create primitive-descriptor
                dnnl::sum::primitive_desc sumPrimitiveDesc(scales, memDesc, *_engine);

                // Add sum primitive to list of primitives
                ELSA_ML_ADD_DNNL_PRIMITIVE(_backwardStream, dnnl::sum(sumPrimitiveDesc));

                // We replace the first output-gradient by the sum of all output-gradients
                _backwardStream.arguments.push_back(
                    {{DNNL_ARG_DST, *_outputGradient.front().effectiveMemory}});
                for (std::size_t i = 0; i < _outputGradient.size(); ++i) {
                    _backwardStream.arguments.back().insert(
                        {DNNL_ARG_MULTIPLE_SRC + i, mem[asUnsigned(i)]});
                }
            }
        }

        template class DnnlLayer<float>;
    } // namespace detail
} // namespace elsa::ml
