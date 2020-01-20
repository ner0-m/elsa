#include "DnnlSequentialNetwork.h"
#include <iostream>

namespace elsa
{
    template <typename data_t>
    DnnlSequentialNetwork<data_t>::DnnlSequentialNetwork(
        const std::vector<std::shared_ptr<Layer<data_t, MlBackend::Dnnl>>>& layerStack)
        : _layerStack(layerStack),
          _engine(std::make_shared<dnnl::engine>(dnnl::engine::kind::cpu, 0)),
          _executionStream(*_engine)
    {
    }

    template <typename data_t>
    void DnnlSequentialNetwork<data_t>::compile(Loss loss)
    {
        if (_layerStack.empty())
            throw std::logic_error("Cannot compile network because it contains no layers");

        // Set engine for all layers
        for (auto&& layer : _layerStack) {
            layer->getBackend()->setEngine(_engine);
            layer->getBackend()->initialize();
        }

        // Set loss
        _loss = DnnlLoss<data_t>::createDnnlLoss(_layerStack.back()->getOutputDescriptor(), loss);
        _loss->setEngine(_engine);
        _isCompiled = true;
    }

    template <typename data_t>
    void DnnlSequentialNetwork<data_t>::forwardPropagate(const DataContainer<data_t>& input)
    {
        if (_layerStack.empty())
            throw std::logic_error("Cannot forward propagate because network contains no layers");
        if (!_isCompiled)
            throw std::logic_error("Cannot forward propagate because network is not yet compiled. "
                                   "Use DnnlSequentialNetwork::compiled()");

        _layerStack.front()->getBackend()->setInput(input);
        _layerStack.front()->getBackend()->compile(PropagationKind::Forward);

        for (std::size_t i = 1; i < _layerStack.size(); ++i) {
            auto backend = _layerStack.at(i)->getBackend();

            // Set layer's source memory (and source memory descriptor) to previous layer's output
            // memory
            auto prevBackend = _layerStack.at(i - 1)->getBackend();

            auto prevOutputMem = prevBackend->getOutputMemory();
            backend->setInputMemory(prevOutputMem);

            // Compile the layer
            backend->compile(PropagationKind::Forward);
        }

        // Perform layer-wise forward-propagation
        for (auto& layer : _layerStack)
            layer->getBackend()->forwardPropagate(_executionStream);

        _executionStream.wait();
    }

    template <typename data_t>
    void
        DnnlSequentialNetwork<data_t>::backwardPropagate(std::shared_ptr<dnnl::memory> lossGradient,
                                                         const DataContainer<data_t>& label)
    {
        if (!lossGradient)
            throw std::invalid_argument("Cannot backward propagate because loss gradient is null");
        if (label.getDataDescriptor() != _layerStack.back()->getOutputDescriptor())
            throw std::invalid_argument("Label does not match network output descriptor");
        if (!_isCompiled)
            throw std::logic_error("Cannot forward propagate because network is not yet compiled. "
                                   "Use DnnlSequentialNetwork::compiled()");

        // The last layer's output gradient is the loss gradient
        _layerStack.back()->getBackend()->setOutputGradientMemory(lossGradient);
        _layerStack.back()->getBackend()->compile(PropagationKind::Backward);

        for (int i = _layerStack.size() - 2; i >= 0; --i) {
            auto backend = _layerStack.at(i)->getBackend();

            // Set layer's output gradient memory to next layer's input gradient memory
            auto nextBackend = _layerStack.at(i + 1)->getBackend();
            auto nextInputGradMem = nextBackend->getInputGradientMemory();
            backend->setOutputGradientMemory(nextInputGradMem);

            // Compile the layer
            backend->compile(PropagationKind::Backward);
        }
        // Construct execution stream for the network
        dnnl::stream execStream(*_engine);

        // Perform layer-wise forward-propagation
        for (auto rbit = std::rbegin(_layerStack), reit = std::rend(_layerStack); rbit != reit;
             ++rbit)
            (*rbit)->getBackend()->backwardPropagate(_executionStream);

        _executionStream.wait();
    }

    template <typename data_t>
    std::vector<data_t> DnnlSequentialNetwork<data_t>::train(const DataContainer<data_t>& input,
                                                             const DataContainer<data_t>& label)
    {
        std::vector<data_t> lossVector;

        // Forward propagate the input
        forwardPropagate(input);

        // Calulcate loss
        _loss->evaluate(_layerStack.back()->getBackend()->getOutput(), label);

        // Add loss to vector we finally going to return
        lossVector.push_back(_loss->getLoss());

        // Backward propagate
        backwardPropagate(_loss->getLossGradientMemory(), label);

        // Update trainable layers
        for (auto&& layer : _layerStack) {
            if (layer->isTrainable()) {
                std::static_pointer_cast<DnnlTrainableLayer<data_t>>(layer->getBackend())
                    ->updateTrainableParameters(0.2e-8f);
            }
        }

        return lossVector;
    }

    template <typename data_t>
    DataContainer<data_t> DnnlSequentialNetwork<data_t>::getOutput() const
    {
        if (_layerStack.empty())
            throw std::logic_error("Cannot get network output because the network has no layers");

        // The network's output is the last layer's output
        return _layerStack.back()->getBackend()->getOutput();
    }

    template class DnnlSequentialNetwork<float>;
} // namespace elsa