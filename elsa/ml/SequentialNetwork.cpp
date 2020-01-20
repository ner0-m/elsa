#include "SequentialNetwork.h"

namespace elsa
{
    template <typename data_t, MlBackend Backend>
    SequentialNetwork<data_t, Backend>::SequentialNetwork(const DataDescriptor& inputDescriptor)
        : _inputDescriptor(inputDescriptor.clone())
    {
        // We currently only support floats
        static_assert(std::is_same_v<data_t, float>,
                      "SequentialNetwork currently only supports data-type float");

        // If we receive 1D input, we artificially add a leading batch dimension of 1
        if (_inputDescriptor->getNumberOfDimensions() == 1) {
            IndexVector_t inputVec(2);
            inputVec << 1, _inputDescriptor->getNumberOfCoefficientsPerDimension()[0];
            _inputDescriptor = DataDescriptor(inputVec).clone();
        }

        // If we receive input > 5 we throw
        if (_inputDescriptor->getNumberOfDimensions() > 5)
            throw std::invalid_argument("Network input descriptor cannot have dimension > 5");
    }

    template <typename data_t, MlBackend Backend>
    DataDescriptor SequentialNetwork<data_t, Backend>::getOutputDescriptor() const
    {
        if (_layerStack.empty())
            throw std::logic_error(
                "Cannot return network output descriptor because network contains no layers");

        return _layerStack.back()->getOutputDescriptor();
    }

    template <typename data_t, MlBackend Backend>
    DataDescriptor SequentialNetwork<data_t, Backend>::getInputDescriptor() const
    {
        return *_inputDescriptor;
    }

    template <typename data_t, MlBackend Backend>
    std::size_t SequentialNetwork<data_t, Backend>::getNumberOfLayers() const
    {
        return _layerStack.size();
    }

    template <typename data_t, MlBackend Backend>
    void SequentialNetwork<data_t, Backend>::forwardPropagate(const DataContainer<data_t>& input)
    {
        if (!_backend)
            throw std::logic_error("Cannot forward propagate because the network has not been "
                                   "compiled. Use SequentialNetwork::compile.");

        _backend->forwardPropagate(input);

        _isPropagated = true;
    }

    template <typename data_t, MlBackend Backend>
    DataContainer<data_t> SequentialNetwork<data_t, Backend>::getOutput() const
    {
        if (_layerStack.empty())
            throw std::logic_error("Cannot get network output: The network contains not layers");

        if (!_isPropagated)
            throw std::logic_error("Cannot get network output: No input has been propagated");

        // The network's output is the last layer's output
        return _backend->getOutput();
    }

    template <typename data_t, MlBackend Backend>
    void SequentialNetwork<data_t, Backend>::compile()
    {
        if (_layerStack.empty()) {
            throw std::logic_error("Cannot compile network: Network contains not layers");
        }
        _backend = std::make_unique<BackendNetworkType>(_layerStack);
        _backend->compile(_loss);
    }

    template <typename data_t, MlBackend Backend>
    SequentialNetwork<data_t, Backend>& SequentialNetwork<data_t, Backend>::setLoss(Loss loss)
    {
        _loss = loss;
        return *this;
    }

    template <typename data_t, MlBackend Backend>
    std::vector<data_t>
        SequentialNetwork<data_t, Backend>::train(const DataContainer<data_t>& input,
                                                  const DataContainer<data_t>& label)
    {
        return _backend->train(input, label);
    }

    template <typename data_t, MlBackend Backend>
    std::shared_ptr<Layer<data_t, Backend>>
        SequentialNetwork<data_t, Backend>::getLayer(const index_t index)
    {
        return _layerStack.at(index);
    }

    // Functions to add network layers

    template <typename data_t, MlBackend Backend>
    SequentialNetwork<data_t, Backend>&
        SequentialNetwork<data_t, Backend>::addDenseLayer(int numNeurons, Initializer initializer)
    {
        return addLayer<DenseLayer<data_t, Backend>>(numNeurons, initializer);
    }

    template <typename data_t, MlBackend Backend>
    SequentialNetwork<data_t, Backend>&
        SequentialNetwork<data_t, Backend>::addPoolingLayer(const IndexVector_t& poolingWindow,
                                                            const IndexVector_t& poolingStride)
    {
        return addLayer<PoolingLayer<data_t, Backend>>(poolingWindow, poolingStride);
    }

    template <typename data_t, MlBackend Backend>
    SequentialNetwork<data_t, Backend>&
        SequentialNetwork<data_t, Backend>::addActivationLayer(Activation activation, data_t alpha,
                                                               data_t beta)
    {
        switch (activation) {
            case Activation::Abs:
                return addLayer<Abs<data_t, Backend>>(alpha, beta);
            case Activation::BoundedRelu:
                return addLayer<BoundedRelu<data_t, Backend>>(alpha, beta);
            case Activation::Elu:
                return addLayer<Elu<data_t, Backend>>(alpha, beta);
            case Activation::Exp:
                return addLayer<Exp<data_t, Backend>>(alpha, beta);
            case Activation::Linear:
                return addLayer<Linear<data_t, Backend>>(alpha, beta);
            case Activation::Gelu:
                return addLayer<Gelu<data_t, Backend>>(alpha, beta);
            case Activation::Logistic:
                return addLayer<Logistic<data_t, Backend>>(alpha, beta);
            case Activation::Relu:
                return addLayer<Relu<data_t, Backend>>(alpha, beta);
            case Activation::SoftRelu:
                return addLayer<SoftRelu<data_t, Backend>>(alpha, beta);
            case Activation::Sqrt:
                return addLayer<Sqrt<data_t, Backend>>(alpha, beta);
            case Activation::Square:
                return addLayer<Square<data_t, Backend>>(alpha, beta);
            case Activation::Swish:
                return addLayer<Swish<data_t, Backend>>(alpha, beta);
            case Activation::Tanh:
                return addLayer<Tanh<data_t, Backend>>(alpha, beta);
            default:
                throw std::invalid_argument("Unkown activation");
        }
    }

    template <typename data_t, MlBackend Backend>
    SequentialNetwork<data_t, Backend>& SequentialNetwork<data_t, Backend>::addSoftmaxLayer()
    {
        return addLayer<SoftmaxLayer<data_t, Backend>>();
    }

    template <typename data_t, MlBackend Backend>
    SequentialNetwork<data_t, Backend>& SequentialNetwork<data_t, Backend>::addConvLayer(
        const DataDescriptor& weightsDescriptor, const IndexVector_t& strideVector,
        const IndexVector_t& paddingVector, Initializer initializer)
    {
        return addLayer<ConvLayer<data_t, Backend>>(weightsDescriptor, strideVector, paddingVector,
                                                    initializer);
    }

    template <typename data_t, MlBackend Backend>
    SequentialNetwork<data_t, Backend>& SequentialNetwork<data_t, Backend>::addConvLayer(
        index_t numFilters, const IndexVector_t& weightsVector, const IndexVector_t& strideVector,
        const IndexVector_t& paddingVector, Initializer initializer)
    {
        return addLayer<ConvLayer<data_t, Backend>>(numFilters, weightsVector, strideVector,
                                                    paddingVector, initializer);
    }

    template <typename data_t, MlBackend Backend>
    SequentialNetwork<data_t, Backend>& SequentialNetwork<data_t, Backend>::addConvLayer(
        index_t numFilters, const IndexVector_t& weightsVector, Initializer initializer)
    {
        return addLayer<ConvLayer<data_t, Backend>>(numFilters, weightsVector, initializer);
    }

    template <typename data_t, MlBackend Backend>
    SequentialNetwork<data_t, Backend>&
        SequentialNetwork<data_t, Backend>::addFixedLayer(const JosephsMethod<data_t>& op)
    {
        return addLayer<FixedLayer<data_t, Backend>>(op);
    }

    template <typename data_t, MlBackend Backend>
    SequentialNetwork<data_t, Backend>&
        SequentialNetwork<data_t, Backend>::addLRNLayer(index_t localSize, data_t alpha,
                                                        data_t beta, data_t k)
    {
        return addLayer<LRNLayer<data_t, Backend>>(localSize, alpha, beta, k);
    }

    template class SequentialNetwork<float, MlBackend::Dnnl>;
} // namespace elsa
