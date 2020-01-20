#pragma once

#include <memory>
#include <vector>

#include "DataDescriptor.h"
#include "DataContainer.h"
#include "DnnlLoss.h"
#include "Layer.h"
#include "DnnlTrainableLayer.h"

#include "dnnl.hpp"

namespace elsa
{

    template <typename data_t>
    class DnnlSequentialNetwork final
    {
    public:
        DnnlSequentialNetwork() = default;
        DnnlSequentialNetwork(
            const std::vector<std::shared_ptr<Layer<data_t, MlBackend::Dnnl>>>& layerStack);

        void compile(Loss loss);

        void forwardPropagate(const DataContainer<data_t>& input);

        DataContainer<data_t> getOutput() const;

        std::vector<data_t> train(const DataContainer<data_t>& input,
                                  const DataContainer<data_t>& label);

    private:
        void backwardPropagate(std::shared_ptr<dnnl::memory> lossGradient,
                               const DataContainer<data_t>& label);
        std::unique_ptr<DnnlLoss<data_t>> _loss;
        bool _isCompiled = false;
        std::vector<std::shared_ptr<Layer<data_t, MlBackend::Dnnl>>> _layerStack;
        std::shared_ptr<dnnl::engine> _engine;
        dnnl::stream _executionStream;
    };
} // namespace elsa