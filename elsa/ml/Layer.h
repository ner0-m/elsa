#pragma once

#include <vector>
#include <memory>

#include "elsaDefines.h"
#include "DataDescriptor.h"
#include "DnnlLayer.h"

namespace elsa
{

    /// enum class containing tag encodings for deep-learning backends
    enum class MlBackend {
        /// Deep Neural Network Library
        Dnnl
    };

    namespace detail
    {
        template <typename Layer>
        struct BackendSelector {
            using Type = std::false_type;
        };
    } // namespace detail

    template <typename data_t, MlBackend Backend>
    class Layer
    {
    public:
        /// The type of the layer's backend implementation
        using BackendLayerBaseType =
            std::conditional_t<Backend == MlBackend::Dnnl, DnnlLayer<data_t>, std::false_type>;

        /// The layer's input descriptor
        const DataDescriptor& getInputDescriptor() const;

        /// The layer's output descriptor
        const DataDescriptor& getOutputDescriptor() const;

        /// Get a vector with pointers to layers that are successors to the current layer
        const auto& getSuccessors() const;

        /// Get a vector with pointers to layers that are predecessor to the current layer
        const auto& getPredecessors() const;

        std::shared_ptr<BackendLayerBaseType> getBackend();

    protected:
        Layer(const DataDescriptor& inputDescriptor);

        Layer(const Layer&) = delete;

        void addSuccessor(std::shared_ptr<Layer<data_t, Backend>> successor);
        void addPredecessor(std::shared_ptr<Layer<data_t, Backend>> predecessor);

        /// DataDescriptor for the layer's input
        std::unique_ptr<DataDescriptor> _inputDescriptor;

        /// DataDescriptor for the layer's output.
        std::unique_ptr<DataDescriptor> _outputDescriptor;

        std::vector<std::shared_ptr<Layer>> _successors;
        std::vector<std::shared_ptr<Layer>> _predecessors;

        std::shared_ptr<BackendLayerBaseType> _backend;
    };

    template <typename data_t, MlBackend Backend>
    Layer<data_t, Backend>::Layer(const DataDescriptor& inputDescriptor)
        : _inputDescriptor(inputDescriptor.clone())
    {
    }

    template <typename data_t, MlBackend Backend>
    inline void
        Layer<data_t, Backend>::addSuccessor(std::shared_ptr<Layer<data_t, Backend>> successor)
    {
        if (!successor)
            throw std::invalid_argument("Pointer to successor layer is null");

        _successors.push_back(successor);
    }

    template <typename data_t, MlBackend Backend>
    inline void
        Layer<data_t, Backend>::addPredecessor(std::shared_ptr<Layer<data_t, Backend>> predecessor)
    {
        if (!predecessor)
            throw std::invalid_argument("Pointer to predecessor layer is null");

        _predecessors.push_back(predecessor);
    }

    template <typename data_t, MlBackend Backend>
    inline const DataDescriptor& Layer<data_t, Backend>::getInputDescriptor() const
    {
        return *_inputDescriptor;
    }

    template <typename data_t, MlBackend Backend>
    inline const DataDescriptor& Layer<data_t, Backend>::getOutputDescriptor() const
    {
        return *_outputDescriptor;
    }

    template <typename data_t, MlBackend Backend>
    inline std::shared_ptr<typename Layer<data_t, Backend>::BackendLayerBaseType>
        Layer<data_t, Backend>::getBackend()
    {
        if (!_backend)
            throw std::logic_error("Missing layer backend");
        return _backend;
    }

} // namespace elsa