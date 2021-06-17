#include "DnnlMerging.h"
#include "TypeCasts.hpp"

namespace elsa::ml
{
    namespace detail
    {
        template <typename data_t>
        DnnlMerging<data_t>::DnnlMerging(const std::vector<VolumeDescriptor>& inputDescriptors,
                                         const VolumeDescriptor& outputDescriptor)
            : DnnlLayer<data_t>(inputDescriptors, outputDescriptor, "DnnlMerging",
                                DnnlLayer<data_t>::anyNumberOfInputs)
        {
        }

        template <typename data_t>
        bool DnnlMerging<data_t>::needsForwardSynchronisation() const
        {
            return true;
        }

        template <typename data_t>
        bool DnnlMerging<data_t>::canMerge() const
        {
            return true;
        }

        template <typename data_t>
        DnnlSum<data_t>::DnnlSum(const std::vector<VolumeDescriptor>& inputDescriptors,
                                 const VolumeDescriptor& outputDescriptor)
            : DnnlMerging<data_t>(inputDescriptors, outputDescriptor)
        {
            // Check that all input-descriptors are equal
            assert(std::adjacent_find(inputDescriptors.begin(), inputDescriptors.end(),
                                      [](const auto& a, const auto& b) { return a != b; })
                       == inputDescriptors.end()
                   && "All input-descriptors for DnnlSum must be equal");
        }

        template <typename data_t>
        void DnnlSum<data_t>::compileForwardStream()
        {
            BaseType::compileForwardStream();

            std::vector<dnnl::memory> mem;
            std::vector<dnnl::memory::desc> memDesc;
            for (std::size_t i = 0; i < _input.size(); ++i) {
                memDesc.push_back(_input[i].descriptor);
                BaseType::validateDnnlMemory(_input[i].effectiveMemory);
                mem.push_back(*_input[i].effectiveMemory);
            }

            // We currently do not support custom scaling since the API does not support it
            std::vector<data_t> scales(_input.size(), data_t(1));

            // Create primitive-descriptor
            _forwardPrimitiveDescriptor = dnnl::sum::primitive_desc(scales, memDesc, *_engine);

            for (std::size_t i = 0; i < _input.size(); ++i) {
                // Reoder input memory if necessary
                this->reorderMemory(_forwardPrimitiveDescriptor.src_desc(), _input[i],
                                    _forwardStream);
            }

            // Add sum primitive to forward-stream
            ELSA_ML_ADD_DNNL_PRIMITIVE(_forwardStream, dnnl::sum(_forwardPrimitiveDescriptor));

            // Allocate output memory
            _output.effectiveMemory =
                std::make_shared<dnnl::memory>(_forwardPrimitiveDescriptor.dst_desc(), *_engine);

            // Validate memory
            BaseType::validateDnnlMemory(_output.effectiveMemory);

            // Add arguments to forward-stream
            _forwardStream.arguments.push_back({{DNNL_ARG_DST, *_output.effectiveMemory}});
            for (std::size_t i = 0; i < _input.size(); ++i) {
                _forwardStream.arguments.back().insert({DNNL_ARG_MULTIPLE_SRC + i, mem[i]});
            }
        }

        template <typename data_t>
        void DnnlSum<data_t>::compileBackwardStream()
        {
            BaseType::compileBackwardStream();

            // Allocate memory for input-gradient if necessary
            for (std::size_t i = 0; i < _inputGradient.size(); ++i) {
                if (!_inputGradient[i].effectiveMemory) {
                    _inputGradient[i].effectiveMemory = std::make_shared<dnnl::memory>(
                        dnnl::memory::desc({{_inputGradient[i].dimensions},
                                            this->_typeTag,
                                            _inputGradient[i].formatTag}),
                        *_engine);
                }
            }
            _outputGradient.front().effectiveMemory = _outputGradient.front().describedMemory;
        }

        template <typename data_t>
        void DnnlSum<data_t>::backwardPropagate([[maybe_unused]] dnnl::stream& executionStream)
        {
            // make sure backward stream has been compiled
            assert(_backwardStream.isCompiled
                   && "Cannot backward propagate because backward-stream has not been compiled");

            // We derive the gradient for a sum layer as follows:
            //
            //
            //      |   ^                   |   ^
            //   i0 |   | dE/di0         i1 |   | dE/di1
            //      v   |                   v   |
            //   +--------------------------------+
            //   |             SUM                |
            //   +--------------------------------+
            //                |    ^
            //              o |    | dE/do
            //                v    |
            //
            // The input-gradient along the path of input i0 is given by
            //    dE/di0 = dE/do * do/di0
            //             ^^^^^   ^^^^^^
            //             |       | i0 as the partial derivative of i0+i1
            //             | output-gradient

            // Get output-gradient memory
            Eigen::Map<Eigen::ArrayX<data_t>> outputGrad(
                static_cast<data_t*>(_outputGradient.front().effectiveMemory->get_data_handle()),
                _outputDescriptor->getNumberOfCoefficients());

            for (int i = 0; i < _inputGradient.size(); ++i) {
                BaseType::validateDnnlMemory(_inputGradient[asUnsigned(i)].effectiveMemory);
                BaseType::validateDnnlMemory(_outputGradient.front().effectiveMemory);
                BaseType::validateDnnlMemory(_input[asUnsigned(i)].effectiveMemory);

                // Get input-gradient memory
                Eigen::Map<Eigen::ArrayX<data_t>> inputGrad(
                    static_cast<data_t*>(
                        _inputGradient[asUnsigned(i)].effectiveMemory->get_data_handle()),
                    _inputDescriptor[asUnsigned(i)]->getNumberOfCoefficients());

                // Get input memory
                Eigen::Map<Eigen::ArrayX<data_t>> input(
                    static_cast<data_t*>(_input[asUnsigned(i)].effectiveMemory->get_data_handle()),
                    _inputDescriptor[asUnsigned(i)]->getNumberOfCoefficients());

                // Compute input-gradient
                inputGrad = outputGrad * input;
            }
        }

        template <typename data_t>
        DnnlConcatenate<data_t>::DnnlConcatenate(
            index_t axis, const std::vector<VolumeDescriptor>& inputDescriptors,
            const VolumeDescriptor& outputDescriptor)
            : DnnlMerging<data_t>(inputDescriptors, outputDescriptor), _axis(axis)
        {

            // Check that all input-descriptors are equal
            assert(std::adjacent_find(inputDescriptors.begin(), inputDescriptors.end(),
                                      [](const auto& a, const auto& b) { return a != b; })
                       == inputDescriptors.end()
                   && "All input-descriptors for DnnlSum must be equal");
        }

        template <typename data_t>
        void DnnlConcatenate<data_t>::compileForwardStream()
        {
            BaseType::compileForwardStream();

            std::vector<dnnl::memory> mem;
            std::vector<dnnl::memory::desc> memDesc;
            for (std::size_t i = 0; i < _input.size(); ++i) {
                memDesc.push_back(_input[i].descriptor);
                BaseType::validateDnnlMemory(_input[i].effectiveMemory);
                mem.push_back(*_input[i].effectiveMemory);
            }

            // Create primitive-descriptor
            _forwardPrimitiveDescriptor = dnnl::concat::primitive_desc(_axis, memDesc, *_engine);

            for (std::size_t i = 0; i < _input.size(); ++i) {
                // Reoder input memory if necessary
                this->reorderMemory(_forwardPrimitiveDescriptor.src_desc(), _input[i],
                                    _forwardStream);
            }

            // Add sum primitive to forward-stream
            ELSA_ML_ADD_DNNL_PRIMITIVE(_forwardStream, dnnl::concat(_forwardPrimitiveDescriptor));

            // Allocate output memory
            _output.effectiveMemory =
                std::make_shared<dnnl::memory>(_forwardPrimitiveDescriptor.dst_desc(), *_engine);

            // Validate memory
            BaseType::validateDnnlMemory(_output.effectiveMemory);

            // Add arguments to forward-stream
            _forwardStream.arguments.push_back({{DNNL_ARG_DST, *_output.effectiveMemory}});
            for (std::size_t i = 0; i < _input.size(); ++i) {
                _forwardStream.arguments.back().insert({DNNL_ARG_MULTIPLE_SRC + i, mem[i]});
            }
        }

        template <typename data_t>
        void DnnlConcatenate<data_t>::compileBackwardStream()
        {
            BaseType::compileBackwardStream();

            // Allocate memory for input-gradient if necessary
            for (std::size_t i = 0; i < _inputGradient.size(); ++i) {
                if (!_inputGradient[i].effectiveMemory) {
                    _inputGradient[i].effectiveMemory = std::make_shared<dnnl::memory>(
                        dnnl::memory::desc({{_inputGradient[i].dimensions},
                                            this->_typeTag,
                                            _inputGradient[i].formatTag}),
                        *_engine);
                }
            }
            _outputGradient.front().effectiveMemory = _outputGradient.front().describedMemory;
        }

        template <typename data_t>
        void DnnlConcatenate<data_t>::backwardPropagate([
            [maybe_unused]] dnnl::stream& executionStream)
        {
            // make sure backward stream has been compiled
            assert(_backwardStream.isCompiled
                   && "Cannot backward propagate because backward-stream has not been compiled");

            // We derive the gradient for a concat layer as follows:
            //
            // If the Concatenate layer receives three inputs i0, i1, i2 with
            // shapes (n, c0, h, w), (n, c1, h, w) and (n, c2, h, w)
            // respectively and c is the concatenation axis, the output has
            // shape (n, c0+c1+c2, h, w).
            //
            // The incoming gradient for the Concatentation layer has then
            // also shape (n, c0+c1+c2, h, w).
            //
            // The gradient for each of the inputs is then the slice of the
            // incoming gradient along c that matches the slice of the input,
            // e.g. i0 gets slice (n, c0, h, w) of the incoming gradient.
        }

        template class DnnlMerging<float>;
        template class DnnlSum<float>;
        template class DnnlConcatenate<float>;
    } // namespace detail
} // namespace elsa::ml
