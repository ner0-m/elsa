#pragma once

#include <string>
#include <sstream>
#include <set>
#include <utility>
#include <vector>

#include "elsaDefines.h"
#include "TypeCasts.hpp"
#include "Common.h"
#include "BackendAdaptor.h"
#include "State.h"
#include "Layer.h"
#include "Input.h"
#include "Optimizer.h"
#include "Loss.h"

#ifdef ELSA_HAS_CUDNN_BACKEND
#include "CudnnDataContainerInterface.h"
#endif

namespace elsa::ml
{
    /// A ml model that can be used for training and inference.
    ///
    /// @author David Tellenbach
    ///
    /// @tparam data_t The type of all coefficients used in the model. This
    /// parameter is optional and defaults to real_t.
    /// @tparam Backend The MlBackend that will be used for inference and
    /// training. This parameter is optional and defaults to MlBackend::Auto.
    template <typename data_t = real_t, MlBackend Backend = MlBackend::Auto>
    class Model
    {
    public:
        /// Default constructor
        Model() = default;

        /// Construct a model by specifying a set of in- and outputs.
        /// @param inputs a list if input layers
        /// @param outputs a list of layers that serve as this model's outputs
        /// @param name a name for this model
        Model(std::initializer_list<Input<data_t>*> inputs,
              std::initializer_list<Layer<data_t>*> outputs, const std::string& name = "model");

        /// Construct a model by specifying a single input and a single output
        ///
        /// @param input an input layer
        /// @param output a layer that serves as this model's output
        /// @param name a name for this model
        Model(Input<data_t>* input, Layer<data_t>* output, const std::string& name = "model");

        /// Compile the model by specifying a loss function and an optimizer.
        void compile(const Loss<data_t>& loss, Optimizer<data_t>* optimizer);

        /// Get a constant reference to the model's loss function.
        const Loss<data_t>& getLoss() const;

        /// Get this model's optimizer.
        Optimizer<data_t>* getOptimizer();

        /// Get this model's batch-size.
        index_t getBatchSize() const;

        /// Get this models name.
        std::string getName() const;

        /// Get a list of this model's inputs.
        std::vector<Input<data_t>*> getInputs();

        /// Get a list of this model's outputs.
        std::vector<Layer<data_t>*> getOutputs();

        detail::Graph<typename detail::BackendSelector<data_t, Backend, LayerType::Undefined>::Type,
                      false>&
            getBackendGraph();

        const detail::Graph<
            typename detail::BackendSelector<data_t, Backend, LayerType::Undefined>::Type, false>&
            getBackendGraph() const;

        struct History {
            std::vector<data_t> loss;
            std::vector<data_t> metric;
        };

        /// Train the model by providing inputs x and labels y.
        ///
        /// @param x model input
        /// @param y labels
        /// @param epochs training epochs
        ///
        /// @returns a Model::History object
        History fit(const std::vector<DataContainer<data_t>>& x,
                    const std::vector<DataContainer<data_t>>& y, index_t epochs);

        /// Perform inference for a given input x.
        DataContainer<data_t> predict(const DataContainer<data_t>& x);

        /// Pretty print this model.
        template <typename T, MlBackend B>
        friend std::ostream& operator<<(std::ostream& os, const Model<T, B>& model);

    protected:
        friend struct detail::BackendAdaptor<data_t, Backend>;

        void setInputDescriptors();

        std::string name_;
        index_t batchSize_;
        std::vector<Input<data_t>*> inputs_;
        std::vector<Layer<data_t>*> outputs_;

        detail::Graph<typename detail::BackendSelector<data_t, Backend, LayerType::Undefined>::Type,
                      false>
            backendGraph_;
        std::set<index_t> backendOutputs_;

        Loss<data_t> loss_;
        Optimizer<data_t>* optimizer_;
    };

    template <typename T, MlBackend B>
    std::ostream& operator<<(std::ostream& os, const Model<T, B>& model)
    {
        // get graph
        auto& graph = detail::State<T>::getGraph();

        os << "Model: " << model.getName() << "\n";
        os << "________________________________________________________________________________\n";
        // get format for output-desc
        Eigen::IOFormat fmt(Eigen::StreamPrecision, 0, ", ", ", ", "", "", "(", ")");
        os << std::left << std::setw(35) << "Layer (type)" << std::setw(20) << "Output Shape"
           << std::setw(10) << "Param #" << std::setw(10) << "Connected to\n";
        os << "================================================================================\n";

        index_t totalNumberOfParams = 0;
        index_t count = 0;
        for (const auto& node : graph.getNodes()) {
            count++;
            bool first = true;
            os << *node.second.getData();
            totalNumberOfParams += node.second.getData()->getNumberOfTrainableParameters();

            for (auto& edge : graph.getOutgoingEdges(node.first)) {
                if (first) {
                    os << edge.end()->getData()->getName() << "\n";
                    first = false;
                } else {
                    os << std::right << std::setw(65 + edge.end()->getData()->getName().size())
                       << edge.end()->getData()->getName() << "\n";
                }
            }
            if (asUnsigned(count) != graph.getNodes().size()) {
                os << "____________________________________________________________________________"
                      "____\n";
            }
        }

        os << "\n================================================================================"
              "\n";
        os << "Total trainable params: " << totalNumberOfParams << "\n";
        os << "________________________________________________________________________________";

        return os;
    }

    template <typename data_t = real_t, MlBackend Backend = MlBackend::Dnnl>
    class Sequential : public Model<data_t, Backend>
    {
    public:
        Sequential() = default;

        ~Sequential()
        {
            for (auto& layer : layers_)
                delete layer;
        }

        template <typename FirstLayerType, typename... LayerTypes>
        Sequential(FirstLayerType&& input, LayerTypes&&... layers)
        {
            addLayerHelper(std::forward<FirstLayerType>(input),
                           std::forward<LayerTypes>(layers)...);
            assert(!layers_.empty());
            this->outputs_.push_back(layers_.back());
            this->batchSize_ = this->inputs_.front()->getBatchSize();
            this->setInputDescriptors();
        }

        template <typename LayerType>
        void add(LayerType&& layer)
        {
            if constexpr (std::is_same_v<std::decay_t<LayerType>, Input<data_t>>) {
                if (layers_.size() != 0) {
                    throw std::invalid_argument(
                        "Only first layer of a sequential model can be Input");
                }
                Input<data_t>* input = new LayerType(std::move(layer));
                this->inputs_.push_back(input);
                layers_.push_back(input);
            } else {
                if (!layers_.size()) {
                    throw std::invalid_argument(
                        "First layer of a sequential model must be an input layer");
                }
                Layer<data_t>* nextLayer = new LayerType(std::move(layer));
                nextLayer->setInput(layers_.back());
                layers_.push_back(nextLayer);
            }
        }

    private:
        template <typename T, typename... Ts>
        void addLayerHelper(T&& first, Ts&&... rest)
        {
            add<T>(std::forward<T>(first));

            if constexpr (sizeof...(rest) > 0) {
                addLayerHelper(std::forward<Ts>(rest)...);
            }
        }

        std::vector<Layer<data_t>*> layers_;
    };
} // namespace elsa::ml
