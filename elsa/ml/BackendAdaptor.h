#pragma once

#include "Common.h"
#include "DataContainer.h"
#include "TypeCasts.hpp"

namespace elsa::ml
{
    template <typename data_t, MlBackend Backend>
    class Model;

    namespace detail
    {
        template <typename data_t, MlBackend Backend, LayerType Layer>
        struct BackendSelector {
            using Type = std::false_type;
        };

#define ELSA_ML_MAKE_BACKEND_LAYER_SELECTOR(Backend, Layer, BackendLayer)  \
    template <typename data_t>                                             \
    struct BackendSelector<data_t, MlBackend::Backend, LayerType::Layer> { \
        using Type = BackendLayer<data_t>;                                 \
    }

        /// Generic BackendAdaptor.
        ///
        /// Backend-specific logic is implemented in template specializations
        /// of this struct.
        template <typename data_t, MlBackend Backend>
        struct BackendAdaptor {
            static void constructBackendGraph([[maybe_unused]] Model<data_t, Backend>*)
            {
                throw std::logic_error("No elsa ML backend available");
            }

            static DataContainer<data_t> predict([[maybe_unused]] Model<data_t, Backend>*,
                                                 [[maybe_unused]] const DataContainer<data_t>&)
            {
                throw std::logic_error("No elsa ML backend available");
            }

            static typename Model<data_t, Backend>::History
                fit([[maybe_unused]] Model<data_t, Backend>*,
                    [[maybe_unused]] const std::vector<DataContainer<data_t>>&,
                    [[maybe_unused]] const std::vector<DataContainer<data_t>>&, index_t)
            {
                throw std::logic_error("No elsa ML backend available");
            }
        };

        /// Attach batch-size to a volume-descriptor
        ///
        /// If a we have a descriptor
        ///   {w, h, c}
        /// this creates a new descriptor
        ///   {w, h, c, n}.
        static inline VolumeDescriptor
            attachBatchSizeToVolumeDescriptor(index_t batchSize, const VolumeDescriptor& desc)
        {
            IndexVector_t dims(desc.getNumberOfDimensions() + 1);
            dims.head(desc.getNumberOfDimensions()) = desc.getNumberOfCoefficientsPerDimension();
            dims.tail(1)[asUnsigned(0)] = batchSize;
            return VolumeDescriptor(dims);
        }

        /// Reverse a volume-descriptor
        ///
        /// If we have a descriptor
        ///   {w, h, c, n}
        /// this creates a descriptor
        ///   {n, c, h, w}.
        static inline VolumeDescriptor reverseVolumeDescriptor(const VolumeDescriptor& desc)
        {
            IndexVector_t dims = desc.getNumberOfCoefficientsPerDimension().reverse();
            return VolumeDescriptor(dims);
        }

        template <typename T>
        static auto getCheckedLayerPtr(T&& node)
        {
            auto layer = node.getData();
            assert(layer != nullptr && "Pointer to backend-layer is null");
            return layer;
        }

        template <typename T>
        static auto getCheckedLayerPtr(T* node)
        {
            auto layer = node->getData();
            assert(layer != nullptr && "Pointer to backend-layer is null");
            return layer;
        }

        template <typename GraphType>
        void setNumberOfOutputGradients(GraphType* backendGraph)
        {
            for (auto&& node : backendGraph->getNodes()) {
                auto layer = getCheckedLayerPtr(node.second);
                layer->setNumberOfOutputGradients(
                    asSigned(backendGraph->getOutgoingEdges(node.first).size()));
            }
        }
    } // namespace detail

} // namespace elsa::ml

#ifdef ELSA_HAS_DNNL_BACKEND
#include "DnnlBackendAdaptor.h"
#endif

#ifdef ELSA_HAS_CUDNN_BACKEND
#include "CudnnBackendAdaptor.h"
#endif
