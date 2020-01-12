#include "DnnlSequentialNetwork.h"

namespace elsa
{
    template <typename data_t>
    DnnlSequentialNetwork<data_t>::DnnlSequentialNetwork(
        std::vector<Layer<data_t, MlBackend::Dnnl>>* layerStack)
        : _layerStack(layerStack),
          _engine(std::make_shared<dnnl::engine>(dnnl::engine::kind::cpu, 0))
    {
    }

    template <typename data_t>
    void DnnlSequentialNetwork<data_t>::compile()
    {
        if (_layerStack->empty())
            throw std::logic_error("Cannot compile network because it contains no layers");

        // Set engine for all layers
        for (auto&& layer : *_layerStack)
            layer.getBackend()->setEngine(_engine);

        _isCompiled = true;
    }

    template <typename data_t>
    void DnnlSequentialNetwork<data_t>::forwardPropagate(const DataContainer<data_t>& input)
    {
        if (_layerStack->empty())
            throw std::logic_error("Cannot forward propagate because network contains no layers");
        if (!_isCompiled)
            throw std::logic_error("Cannot forward propagate because network is not yet compiled. "
                                   "Use DnnlSequentialNetwork::compiled()");

        _layerStack->front().getBackend()->setInput(input);
        _layerStack->front().getBackend()->compile();

        for (std::size_t i = 1; i < _layerStack->size(); ++i) {
            auto backend = _layerStack->at(i).getBackend();

            // Set layer's source memory (and source memory descriptor) to previous layer's output
            // memory
            auto prevBackend = _layerStack->at(i - 1).getBackend();
            auto prevOutputMem = prevBackend->getOutputMemory();
            backend->setSourceMemory(prevOutputMem);

            // Compile the layer
            backend->compile();
        }

        // Construct execution stream for the network
        dnnl::stream execStream(*_engine);

        // Perform layer-wise forward-propagation
        for (std::size_t i = 0; i < _layerStack->size(); ++i) {
            _layerStack->at(i).getBackend()->forwardPropagate(execStream);
        }

        execStream.wait();
    }

    template <typename data_t>
    DataContainer<data_t> DnnlSequentialNetwork<data_t>::getOutput() const
    {
        if (_layerStack->empty())
            throw std::logic_error("Cannot get network output because the network has no layers");

        // The network's output is the last layer's output
        return _layerStack->back().getBackend()->getOutput();
    }

    template class DnnlSequentialNetwork<float>;
} // namespace elsa