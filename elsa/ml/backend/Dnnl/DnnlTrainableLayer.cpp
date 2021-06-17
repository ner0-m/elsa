#include "DnnlTrainableLayer.h"
#include "TypeCasts.hpp"

namespace elsa::ml
{
    namespace detail
    {
        template <typename data_t>
        DnnlTrainableLayer<data_t>::DnnlTrainableLayer(const VolumeDescriptor& inputDescriptor,
                                                       const VolumeDescriptor& outputDescriptor,
                                                       const VolumeDescriptor& weightsDescriptor,
                                                       Initializer initializer)
            : DnnlLayer<data_t>(inputDescriptor, outputDescriptor, "DnnlTrainableLayer"),
              _weightsGradientAcc(asUnsigned(weightsDescriptor.getNumberOfCoefficients())),
              _weightsDescriptor(weightsDescriptor.clone()),
              _initializer(initializer)
        {

            _input.front().canBeReordered = true;

            // Set the layer's fan-in and fan-out. This is needed for random initialization of
            // weights and biases
            _fanInOut.first = inputDescriptor.getNumberOfCoefficients();
            _fanInOut.second = outputDescriptor.getNumberOfCoefficients();

            // Set weights meta information
            for (const auto& dim : weightsDescriptor.getNumberOfCoefficientsPerDimension())
                _weights.dimensions.push_back(dim);

            _weightsGradient.dimensions = _weights.dimensions;

            _weights.formatTag =
                BaseType::dataDescriptorToDnnlMemoryFormatTag(weightsDescriptor,
                                                              /* No input but weights tag */ false);

            _weightsGradient.formatTag = _weights.formatTag;

            _weights.descriptor =
                dnnl::memory::desc({_weights.dimensions}, _typeTag, dnnl::memory::format_tag::any);

            _weightsGradient.descriptor = _weights.descriptor;

            _weightsGradientAcc.setZero();

            IndexVector_t biasVec(1);
            biasVec << _weights.dimensions[0];

            _biasDescriptor = VolumeDescriptor(biasVec).clone();

            // Set weights bias information
            _bias.dimensions.push_back(_weights.dimensions[0]);

            _biasGradient.dimensions = _bias.dimensions;

            _bias.descriptor =
                dnnl::memory::desc({_bias.dimensions}, _typeTag, dnnl::memory::format_tag::any);

            _biasGradient.descriptor = _bias.descriptor;

            _bias.formatTag = dnnl::memory::format_tag::x;

            _biasGradient.formatTag = _bias.formatTag;

            _biasGradientAcc.setZero(_biasDescriptor->getNumberOfCoefficients());

            initialize();
        }

        template <typename data_t>
        void DnnlTrainableLayer<data_t>::setWeights(const DataContainer<data_t>& weights)
        {
            this->writeToDnnlMemory(weights, *_weights.describedMemory);
        }

        template <typename data_t>
        void DnnlTrainableLayer<data_t>::setBias(const DataContainer<data_t>& bias)
        {
            this->writeToDnnlMemory(bias, *_bias.describedMemory);
        }

        template <typename data_t>
        void DnnlTrainableLayer<data_t>::initialize()
        {
            // Construct weights memory and initialize it
            auto weightsDesc =
                dnnl::memory::desc({_weights.dimensions}, _typeTag, _weights.formatTag);
            _weights.describedMemory = std::make_shared<dnnl::memory>(weightsDesc, *_engine);

            InitializerImpl<data_t>::initialize(
                static_cast<data_t*>(_weights.describedMemory->get_data_handle()),
                _weightsDescriptor->getNumberOfCoefficients(), _initializer, _fanInOut);

            // Construct bias memory and initialize it with zero
            auto biasDesc = dnnl::memory::desc({_bias.dimensions}, _typeTag, _bias.formatTag);
            _bias.describedMemory = std::make_shared<dnnl::memory>(biasDesc, *_engine);

            InitializerImpl<data_t>::initialize(
                static_cast<data_t*>(_bias.describedMemory->get_data_handle()), _bias.dimensions[0],
                Initializer::Zeros, _fanInOut);

            // Bias can never be reordered
            _bias.effectiveMemory = _bias.describedMemory;
        }

        template <typename data_t>
        void DnnlTrainableLayer<data_t>::compileForwardStream()
        {
            BaseType::compileForwardStream();
        }

        template <typename data_t>
        void DnnlTrainableLayer<data_t>::compileBackwardStream()
        {
            BaseType::compileBackwardStream();

            // Construct weights memory descriptor and allocate weights memory
            auto weightsDesc = dnnl::memory::desc({_weightsGradient.dimensions}, _typeTag,
                                                  _weightsGradient.formatTag);
            _weightsGradient.describedMemory =
                std::make_shared<dnnl::memory>(weightsDesc, *_engine);

            // Construct bias memory descriptor and allocate bias memory
            auto biasDesc =
                dnnl::memory::desc({_biasGradient.dimensions}, _typeTag, _biasGradient.formatTag);
            _biasGradient.describedMemory = std::make_shared<dnnl::memory>(biasDesc, *_engine);

            // Bias can never be reordered
            _biasGradient.effectiveMemory = _biasGradient.describedMemory;
        }

        template <typename data_t>
        DataContainer<data_t> DnnlTrainableLayer<data_t>::getGradientWeights() const
        {
            DataContainer<data_t> output(*_weightsDescriptor);
            this->readFromDnnlMemory(output, *_weightsGradient.effectiveMemory);
            return output;
        }

        template <typename data_t>
        DataContainer<data_t> DnnlTrainableLayer<data_t>::getGradientBias() const
        {
            DataContainer<data_t> output(*_biasDescriptor);
            this->readFromDnnlMemory(output, *_biasGradient.effectiveMemory);
            return output;
        }

        template <typename data_t>
        void DnnlTrainableLayer<data_t>::updateTrainableParameters()
        {
            index_t batchSize =
                this->_inputDescriptor.front()->getNumberOfCoefficientsPerDimension()[0];

            // Update weights
            _weightsGradientAcc /= static_cast<data_t>(batchSize);
            weightsOptimizer_->updateParameter(
                _weightsGradientAcc.data(), batchSize,
                static_cast<data_t*>(_weights.effectiveMemory->get_data_handle()));

            // Update bias
            _biasGradientAcc /= static_cast<data_t>(batchSize);
            biasOptimizer_->updateParameter(
                _biasGradientAcc.data(), batchSize,
                static_cast<data_t*>(_bias.effectiveMemory->get_data_handle()));

            // Reset accumulated gradient
            _weightsGradientAcc.setZero();
            _biasGradientAcc.setZero();
        }

        template <typename data_t>
        void DnnlTrainableLayer<data_t>::accumulatedGradients()
        {
            // Accumulate weights
            Eigen::Map<Eigen::ArrayX<data_t>> weightsGradientMem(
                static_cast<data_t*>(_weightsGradient.effectiveMemory->get_data_handle()),
                _weightsDescriptor->getNumberOfCoefficients());

            assert(_weightsGradientAcc.size() == weightsGradientMem.size()
                   && "Size of accumulated weigths must match size of weights");

            _weightsGradientAcc += weightsGradientMem;

            // Accumulate bias
            Eigen::Map<Eigen::ArrayX<data_t>> biasGradientMem(
                static_cast<data_t*>(_biasGradient.effectiveMemory->get_data_handle()),
                _biasDescriptor->getNumberOfCoefficients());

            assert(_biasGradientAcc.size() == biasGradientMem.size()
                   && "Size of accumulated bias must match size of bias");
            _biasGradientAcc += biasGradientMem;
        }

        template <typename data_t>
        void DnnlTrainableLayer<data_t>::backwardPropagate(dnnl::stream& executionStream)
        {
            // Backward propagate as usual
            BaseType::backwardPropagate(executionStream);

            // Accumulate propagated gradients
            accumulatedGradients();
        }

        template <typename data_t>
        bool DnnlTrainableLayer<data_t>::isTrainable() const
        {
            return true;
        }

        template class DnnlTrainableLayer<float>;

    } // namespace detail
} // namespace elsa::ml
