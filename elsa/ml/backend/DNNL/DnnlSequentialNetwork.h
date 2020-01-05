#pragma once

#include <memory>
#include <vector>

#include "DataDescriptor.h"
#include "DataContainer.h"
#include "Layer.h"

#include "dnnl.hpp"

namespace elsa
{
    template <typename data_t>
    class DnnlSequentialNetwork final
    {
    public:
        DnnlSequentialNetwork() = default;
        DnnlSequentialNetwork(std::vector<Layer<data_t, MlBackend::Dnnl>>* layerStack);

        void compile();

        void forwardPropagate(const DataContainer<data_t>& input);

        DataContainer<data_t> getOutput() const;

    private:
        bool _isCompiled = false;
        std::vector<Layer<data_t, MlBackend::Dnnl>>* _layerStack;
        std::shared_ptr<dnnl::engine> _engine;
    };
} // namespace elsa