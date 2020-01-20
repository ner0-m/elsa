#include "DnnlFixedLayer.h"
#include <iostream>
#include "RandomInitializer.h"

namespace elsa
{
    template <typename data_t>
    DnnlFixedLayer<data_t>::DnnlFixedLayer(const DataDescriptor& inputDescriptor,
                                           const DataDescriptor& outputDescriptor,
                                           const JosephsMethod<data_t>& op)
        : DnnlLayer<data_t>(inputDescriptor, outputDescriptor), _operator(op.clone())
    {

        std::unique_ptr<DataDescriptor> operatorOutputDesc;
        // Josephs method supports 2d and 3d only, so we initialize it accordingly.
        // Since a forward propagation of the layer is a applyAdjoint, rather than
        // a apply of the operator, we swap input- and output-descriptors
        if (_input.formatTag == dnnl::memory::format_tag::nc) {
            _operatorOutputDescriptor = inputDescriptor.clone();
        }
        if (_input.formatTag == dnnl::memory::format_tag::nchw) {
            IndexVector_t vec(2);
            vec << inputDescriptor.getNumberOfCoefficientsPerDimension()[2],
                inputDescriptor.getNumberOfCoefficientsPerDimension()[3];
            _operatorOutputDescriptor = DataDescriptor(vec).clone();
        }
        if (_input.formatTag == dnnl::memory::format_tag::ncdhw) {
            IndexVector_t vec(3);
            vec << inputDescriptor.getNumberOfCoefficientsPerDimension()[2],
                inputDescriptor.getNumberOfCoefficientsPerDimension()[3],
                inputDescriptor.getNumberOfCoefficientsPerDimension()[4];
            _operatorOutputDescriptor = DataDescriptor(vec).clone();
        }
        // TODO: Choose the correct operator input descriptor
        _operatorInputDescriptor = outputDescriptor.clone();
    }

    template <typename data_t>
    void DnnlFixedLayer<data_t>::initialize()
    {
        if (!_isInitialized) {
            // if (!_input.describedMemory) {
            //     _input.describedMemory = std::make_shared<dnnl::memory>(
            //         dnnl::memory::desc({_input.dimensions}, _typeTag, _input.formatTag),
            //         *_engine);
            // }
            // _input.effectiveMemory = _input.describedMemory;

            if (!_output.describedMemory) {
                _output.describedMemory = std::make_shared<dnnl::memory>(
                    dnnl::memory::desc({_output.dimensions}, _typeTag, _output.formatTag),
                    *_engine);
            }
            _output.effectiveMemory = _output.describedMemory;

            if (!_inputGradient.describedMemory) {
                _inputGradient.describedMemory = std::make_shared<dnnl::memory>(
                    dnnl::memory::desc({_inputGradient.dimensions}, _typeTag,
                                       _inputGradient.formatTag),
                    *_engine);
            }
            _inputGradient.effectiveMemory = _inputGradient.describedMemory;

            if (!_outputGradient.describedMemory) {
                _outputGradient.describedMemory = std::make_shared<dnnl::memory>(
                    dnnl::memory::desc({_outputGradient.dimensions}, _typeTag,
                                       _outputGradient.formatTag),
                    *_engine);
            }
            _outputGradient.effectiveMemory = _outputGradient.describedMemory;
            _isInitialized = true;
        }
    }

    static void validateDnnlMemory(std::shared_ptr<dnnl::memory> mem)
    {
        assert(mem && "Pointer to memory cannot be null");
        assert(mem->get_desc().get_size() && "Memory cannot have size 0,");
    }
#define IS_ZERO(container)                                           \
    if (std::all_of(container.begin(), container.end(),              \
                    [](const auto& coeff) { return coeff == 0.f; })) \
        std::cout << #container << " has only 0 coeffs\n";

    template <typename data_t>
    void DnnlFixedLayer<data_t>::forwardPropagate([[maybe_unused]] dnnl::stream& executionStream)
    {
        validateDnnlMemory(_input.describedMemory);
        DataContainer<float> inputContainer(*_inputDescriptor);
        this->readFromDnnlMemory(inputContainer, *_input.describedMemory);

        IS_ZERO(inputContainer);

        auto outputContainer = _operator->applyAdjoint(inputContainer);
        validateDnnlMemory(_output.effectiveMemory);
        this->writeToDnnlMemory(outputContainer, *_output.effectiveMemory);
        // if (_input.formatTag == dnnl::memory::format_tag::nc) {
        //     auto outputContainer = _operator->applyAdjoint(inputContainer);
        //     validateDnnlMemory(_output.effectiveMemory);
        //     this->writeToDnnlMemory(outputContainer, *_output.effectiveMemory);
        // }
        // // Since JosephsMethod accepts 2d or 3d inputs only, we have to split
        // // up possible further dimensions
        // if (_input.formatTag == dnnl::memory::format_tag::nchw) {
        //     auto N = _inputDescriptor->getNumberOfCoefficientsPerDimension()[0];
        //     auto C = _inputDescriptor->getNumberOfCoefficientsPerDimension()[1];
        //     auto H = _inputDescriptor->getNumberOfCoefficientsPerDimension()[2];
        //     auto W = _inputDescriptor->getNumberOfCoefficientsPerDimension()[3];

        //     DataContainer<data_t> outputContainer(*_outputDescriptor);

        //     for (int n = 0; n < N; ++n) {
        //         auto inputSlice = DataContainer<data_t>(*_operatorOutputDescriptor);
        //         for (int h = 0; h < H; ++h) {
        //             for (int w = 0; w < W; ++w) {
        //                 inputSlice(h, w) = inputContainer(n, 0, h, w);
        //             }
        //         }
        //         auto outputSlice = _operator->applyAdjoint(inputSlice);
        //         for (int h = 0; h < H; ++h) {
        //             for (int w = 0; w < W; ++w) {
        //                 outputContainer(n, 0, h, w) = outputSlice(h, w);
        //             }
        //         }
        //     }
        //     validateDnnlMemory(_output.effectiveMemory);
        //     this->writeToDnnlMemory(outputContainer, *_output.effectiveMemory);
        //     IS_ZERO(outputContainer);
        // }
    }

    template <typename data_t>
    void DnnlFixedLayer<data_t>::backwardPropagate([[maybe_unused]] dnnl::stream& executionStream)
    {
        validateDnnlMemory(_outputGradient.describedMemory);

        DataContainer<float> outputGradientContainer(*_outputDescriptor);
        this->readFromDnnlMemory(outputGradientContainer, *_outputGradient.describedMemory);
        IS_ZERO(outputGradientContainer);
        auto inputGradientContainer = _operator->apply(outputGradientContainer);
        validateDnnlMemory(_inputGradient.effectiveMemory);

        this->writeToDnnlMemory(inputGradientContainer, *_inputGradient.effectiveMemory);

        //         if (_input.formatTag == dnnl::memory::format_tag::nchw) {
        //             auto N = _outputDescriptor->getNumberOfCoefficientsPerDimension()[0];
        //             auto C = _outputDescriptor->getNumberOfCoefficientsPerDimension()[1];
        //             auto H = _outputDescriptor->getNumberOfCoefficientsPerDimension()[2];
        //             auto W = _outputDescriptor->getNumberOfCoefficientsPerDimension()[3];

        //             DataContainer<data_t> inputGradientContainer(*_inputDescriptor);

        // #pragma omp parallel for
        //             for (int n = 0; n < N; ++n) {
        //                 auto outputGradientSlice =
        //                 DataContainer<data_t>(*_operatorInputDescriptor);

        //                 for (int h = 0; h < H; ++h) {
        //                     for (int w = 0; w < W; ++w) {
        //                         outputGradientSlice(h, w) = outputGradientContainer(n, 0, h, w);
        //                     }
        //                 }
        //                 auto inputGradientSlice = _operator->apply(outputGradientSlice);

        //                 for (int h = 0; h < H; ++h) {
        //                     for (int w = 0; w < W; ++w) {
        //                         inputGradientContainer(n, 0, h, w) = inputGradientSlice(h, w);
        //                     }
        //                 }
        //             }
        //             validateDnnlMemory(_inputGradient.effectiveMemory);

        //             this->writeToDnnlMemory(inputGradientContainer,
        //             *_inputGradient.effectiveMemory);
        //         }
    }

    template <typename data_t>
    void DnnlFixedLayer<data_t>::compileForwardStream()
    {
    }

    template <typename data_t>
    void DnnlFixedLayer<data_t>::compileBackwardStream()
    {
    }

    template class DnnlFixedLayer<float>;

} // namespace elsa
