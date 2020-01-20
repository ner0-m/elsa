#pragma once

#include <vector>
#include <memory>

#include "elsaDefines.h"
#include "DataDescriptor.h"
#include "DataContainer.h"
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

        virtual bool isTrainable() const;

        virtual bool isOperator() const;

        virtual ~Layer() = default;

        Layer() = default;

        Layer(Layer&& other) = default;

        Layer<data_t, Backend>& operator=(Layer&& other) = default;

        /// The layer's input descriptor
        const DataDescriptor& getInputDescriptor() const;

        /// The layer's output descriptor
        const DataDescriptor& getOutputDescriptor() const;

        std::shared_ptr<BackendLayerBaseType> getBackend();

    protected:
        explicit Layer(const DataDescriptor& inputDescriptor);

        /// DataDescriptor for the layer's input
        std::unique_ptr<DataDescriptor> _inputDescriptor;

        /// DataDescriptor for the layer's output.
        std::unique_ptr<DataDescriptor> _outputDescriptor;

        std::shared_ptr<BackendLayerBaseType> _backend;
    };

    template <typename data_t, MlBackend Backend>
    bool Layer<data_t, Backend>::isTrainable() const
    {
        return false;
    }

    template <typename data_t, MlBackend Backend>
    bool Layer<data_t, Backend>::isOperator() const
    {
        return false;
    }

    template <typename data_t, MlBackend Backend>
    Layer<data_t, Backend>::Layer(const DataDescriptor& inputDescriptor)
        : _inputDescriptor(inputDescriptor.clone())
    {
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
