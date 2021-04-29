#pragma once

#include <vector>
#include <list>
#include <memory>
#include <set>
#include <string>

#include "elsaDefines.h"
#include "Common.h"
#include "State.h"
#include "Layer.h"
#include "Dense.h"
#include "Conv.h"
#include "Pooling.h"
#include "ProgressBar.h"
#include "Utils.h"
#include "IdenticalBlocksDescriptor.h"

#include "DnnlNoopLayer.h"
#include "DnnlDenseLayer.h"
#include "DnnlConvolution.h"
#include "DnnlPoolingLayer.h"
#include "DnnlFlattenLayer.h"
#include "DnnlSoftmaxLayer.h"
#include "DnnlMerging.h"
#include "DnnlActivationLayer.h"

namespace elsa::ml
{
    namespace detail
    {
        ELSA_ML_MAKE_BACKEND_LAYER_SELECTOR(Dnnl, Undefined, DnnlLayer);
        ELSA_ML_MAKE_BACKEND_LAYER_SELECTOR(Dnnl, Dense, DnnlDenseLayer);
        ELSA_ML_MAKE_BACKEND_LAYER_SELECTOR(Dnnl, Softmax, DnnlSoftmaxLayer);
        ELSA_ML_MAKE_BACKEND_LAYER_SELECTOR(Dnnl, Conv1D, DnnlConvolution);
        ELSA_ML_MAKE_BACKEND_LAYER_SELECTOR(Dnnl, Conv2D, DnnlConvolution);
        ELSA_ML_MAKE_BACKEND_LAYER_SELECTOR(Dnnl, Conv3D, DnnlConvolution);

        template <typename data_t, typename GraphType>
        static void addActivationNode(index_t index, Activation activation,
                                      const VolumeDescriptor& inputDescriptor, GraphType* graph)
        {
            switch (activation) {
                case Activation::Identity:
                    graph->setData(index, std::make_shared<DnnlNoopLayer<data_t>>(inputDescriptor));
                    break;
                case Activation::Sigmoid:
                    graph->setData(index, std::make_shared<DnnlLogistic<data_t>>(inputDescriptor));
                    break;
                case Activation::Relu:
                    graph->setData(index, std::make_shared<DnnlRelu<data_t>>(inputDescriptor));
                    break;
                case Activation::Tanh:
                    graph->setData(index, std::make_shared<DnnlTanh<data_t>>(inputDescriptor));
                    break;
                case Activation::Elu:
                    graph->setData(index, std::make_shared<DnnlElu<data_t>>(inputDescriptor));
                    break;
                default:
                    assert(false && "This execution path of the code should never be reached");
            }
        }

        template <typename GraphType, typename data_t>
        static void addNodeToBackendGraph(index_t batchSize, GraphType* graph,
                                          const Layer<data_t>* node)
        {
            const index_t nodeIdx = node->getGlobalIndex();

            VolumeDescriptor inputDescriptorWithBatchSize = reverseVolumeDescriptor(
                attachBatchSizeToVolumeDescriptor(batchSize, node->getInputDescriptor()));

            VolumeDescriptor outputDescriptorWithBatchSize = reverseVolumeDescriptor(
                attachBatchSizeToVolumeDescriptor(batchSize, node->getOutputDescriptor()));

            switch (node->getLayerType()) {
                case LayerType::Input: {
                    graph->setData(nodeIdx, std::make_shared<DnnlNoopLayer<data_t>>(
                                                /* input-descriptor */
                                                inputDescriptorWithBatchSize));
                    break;
                }
                case LayerType::Flatten: {
                    graph->setData(nodeIdx, std::make_shared<DnnlFlattenLayer<data_t>>(
                                                /* input-descriptor */
                                                inputDescriptorWithBatchSize,
                                                /* output-descriptor */
                                                outputDescriptorWithBatchSize));
                    break;
                }
                case LayerType::Relu:
                case LayerType::ClippedRelu:
                case LayerType::Sigmoid:
                case LayerType::Tanh:
                case LayerType::Elu: {
                    auto downcastedLayer = dynamic_cast<const ActivationBase<data_t>*>(node);
                    addActivationNode<data_t>(nodeIdx, downcastedLayer->getActivation(),
                                              inputDescriptorWithBatchSize, graph);
                    break;
                }
                case LayerType::Dense: {
                    // Downcast to the polymorphic layer type
                    auto downcastedLayer = dynamic_cast<const Dense<data_t>*>(node);

                    assert(node->getInputDescriptor().getNumberOfDimensions() == 1
                           && "Dense layer requires 1D input");

                    // Build weights descriptor
                    IndexVector_t weightsDims{
                        {downcastedLayer->getNumberOfUnits(),
                         node->getInputDescriptor().getNumberOfCoefficients()}};
                    VolumeDescriptor weightsDescriptor(weightsDims);

                    // Add to backend-graph
                    graph->setData(nodeIdx, std::make_shared<DnnlDenseLayer<data_t>>(
                                                /* input-descriptor */
                                                inputDescriptorWithBatchSize,
                                                /* output-descriptor */
                                                outputDescriptorWithBatchSize,
                                                /* weights-descriptor */
                                                weightsDescriptor,
                                                /* kernel-initializer */
                                                downcastedLayer->getKernelInitializer()));
                    break;
                }
                case LayerType::Softmax: {
                    // We don't need any specific information from the front-end softmax
                    // layer so we just use the base type Layer* and don't have to downcast
                    graph->setData(node->getGlobalIndex(),
                                   std::make_shared<DnnlSoftmaxLayer<data_t>>(
                                       /* input-descriptor */
                                       inputDescriptorWithBatchSize,
                                       /* output-descriptor */
                                       outputDescriptorWithBatchSize));
                    break;
                }
                case LayerType::Conv1D:
                case LayerType::Conv2D:
                case LayerType::Conv3D: {
                    // Downcast to the polymorphic layer type
                    auto downcastedLayer = dynamic_cast<const Conv<data_t>*>(node);

                    // Build weights descriptor
                    VolumeDescriptor filterDescriptor = reverseVolumeDescriptor(
                        attachBatchSizeToVolumeDescriptor(downcastedLayer->getNumberOfFilters(),
                                                          downcastedLayer->getFilterDescriptor()));

                    IndexVector_t strideVector(
                        downcastedLayer->getFilterDescriptor().getNumberOfDimensions() - 1);
                    strideVector.fill(downcastedLayer->getStrides());

                    // TODO(tellenbach): Handle padding correctly
                    IndexVector_t paddingVector(
                        downcastedLayer->getFilterDescriptor().getNumberOfDimensions() - 1);
                    paddingVector.fill(0);

                    // Add to backend-graph
                    graph->setData(nodeIdx, std::make_shared<DnnlConvolution<data_t>>(
                                                /* input-descriptor */
                                                inputDescriptorWithBatchSize,
                                                /* output-descriptor */
                                                outputDescriptorWithBatchSize,
                                                /* weights-descriptor */
                                                filterDescriptor,
                                                /* strides */
                                                strideVector,
                                                /* padding high */
                                                paddingVector,
                                                /* padding low */
                                                paddingVector,
                                                /* kernel-initializer */
                                                downcastedLayer->getKernelInitializer()));
                    break;
                }
                case LayerType::MaxPooling1D:
                case LayerType::MaxPooling2D: {
                    // Downcast to the polymorphic layer type
                    auto downcastedLayer = dynamic_cast<const Pooling<data_t>*>(node);

                    IndexVector_t poolingWindow = downcastedLayer->getPoolSize();
                    IndexVector_t strides(poolingWindow.size());
                    strides.fill(downcastedLayer->getStrides());

                    graph->setData(nodeIdx, std::make_shared<DnnlPoolingLayer<data_t>>(
                                                /* input-descriptor */
                                                inputDescriptorWithBatchSize,
                                                /* output-descriptor */
                                                outputDescriptorWithBatchSize,
                                                /* pooling-window */
                                                poolingWindow,
                                                /* strides */
                                                strides));
                    break;
                }
                case LayerType::Sum: {
                    std::vector<VolumeDescriptor> inputDesc;
                    for (int i = 0; i < node->getNumberOfInputs(); ++i) {
                        inputDesc.push_back(
                            reverseVolumeDescriptor(attachBatchSizeToVolumeDescriptor(
                                batchSize, node->getInputDescriptor(i))));
                    }
                    graph->setData(nodeIdx, std::make_shared<DnnlSum<data_t>>(
                                                /* input-descriptor */
                                                inputDesc,
                                                /* output-descriptor */
                                                outputDescriptorWithBatchSize));
                    break;
                }

                default:
                    throw std::logic_error("Layer of type "
                                           + getEnumMemberAsString(node->getLayerType())
                                           + " is not available for Dnnl backend");
                    // assert(false && "This execution path of the code should never be reached");
            }
        }

        template <typename data_t, typename GraphType>
        static void appendActivation(index_t batchSize, GraphType* graph,
                                     std::set<index_t>* outputs)
        {
            // The API layers allow to specify activation function during the
            // construction of a trainable layer. Dnnl handles activation layers
            // (at least for training, inference allows to fusing of activation
            // function) as separate layers. We therefore insert a separate
            // activation layer after every trainable layer in the backend-graph
            auto& nodes = graph->getNodes();
            for (auto& node : nodes) {
                auto layer = getCheckedLayerPtr(node.second);
                if (layer->isTrainable()) {
                    const index_t nodeIdx = node.first;

                    // Get the corresponding node in the front-end graph as
                    // a trainable layer
                    auto trainableLayer = dynamic_cast<Trainable<data_t>*>(
                        State<data_t>::getGraph().getData(nodeIdx));
                    Activation activation = trainableLayer->getActivation();

                    // Insert a new node that will hold our activation layer
                    const index_t newIdx = nodeIdx + 1;
                    graph->insertNodeAfter(nodeIdx, newIdx);

                    // Input and output are the same for activation layers
                    VolumeDescriptor inputOutputDescriptorWithBatchSize =
                        reverseVolumeDescriptor(attachBatchSizeToVolumeDescriptor(
                            batchSize, trainableLayer->getOutputDescriptor()));

                    // Add the activation layer
                    addActivationNode<data_t>(newIdx, activation,
                                              inputOutputDescriptorWithBatchSize, graph);

                    // If the trainable layer we just handled was one of the
                    // model's output layers, we have to remove it from the
                    // list of outputs and add the newly added activation layer
                    if (outputs->find(nodeIdx) != outputs->end()) {
                        outputs->erase(nodeIdx);
                        outputs->insert(newIdx);
                    }
                }
            }
        }

        template <typename GraphType>
        static void setDnnlEngine(GraphType* graph)
        {
            // Construct Dnnl CPU-engine
            auto engine = std::make_shared<dnnl::engine>(dnnl::engine::kind::cpu, 0);

            // Set engine for all backend layer's
            for (auto&& node : graph->getNodes()) {
                auto layer = getCheckedLayerPtr(node.second);
                layer->setEngine(engine);
            }
        }

        template <typename GraphType, typename data_t>
        static void initializeTrainableParameters(GraphType* graph, Optimizer<data_t>* optimizer)
        {
            // Initialize all trainable parameters of a layer. This e.g.
            // initializes weights and biases
            auto& nodes = graph->getNodes();
            for (auto& node : nodes) {
                auto layer = getCheckedLayerPtr(node.second);
                if (layer->isTrainable()) {
                    auto trainableLayer =
                        std::dynamic_pointer_cast<DnnlTrainableLayer<data_t>>(layer);
                    trainableLayer->setOptimizer(optimizer);
                }
            }
        }

        template <typename data_t>
        struct BackendAdaptor<data_t, MlBackend::Dnnl> {
            static void constructBackendGraph(Model<data_t, MlBackend::Dnnl>* model)
            {
                // Get global layer graph
                auto& graph = State<data_t>::getGraph();

                // Get backend-graph
                auto& backendGraph = model->getBackendGraph();

                // Get the batch size
                index_t batchSize = model->getBatchSize();

                // Get all edges of the front-end graph
                auto& edges = graph.getEdges();

                // For each edge in the front-end graph, add a corresponding
                // edge to the backend-graph
                for (auto&& edges : edges) {
                    auto beginNode = edges.begin()->getData();
                    index_t beginIdx = beginNode->getGlobalIndex();

                    auto endNode = edges.end()->getData();
                    index_t endIdx = endNode->getGlobalIndex();

                    backendGraph.addEdge(beginIdx, endIdx);
                }

                // Get nodes of the front-end graph
                auto& nodes = graph.getNodes();

                // Set backend-layer for each node in the backend-graph
                for (const auto& node : nodes) {
                    auto layer = getCheckedLayerPtr(node.second);
                    index_t idx = layer->getGlobalIndex();
                    if (!backendGraph.getData(idx))
                        addNodeToBackendGraph(batchSize, &backendGraph, layer);
                }

                // Set backend outputs
                for (const auto& output : model->getOutputs())
                    model->backendOutputs_.insert(output->getGlobalIndex());

                // Add a separate activation-layer after each trainable layer
                appendActivation<data_t>(batchSize, &backendGraph, &model->backendOutputs_);

                // Create Dnnl engine and set for all backend-layers
                setDnnlEngine(&backendGraph);

                // Initialize all trainable parameters and set optimizer
                initializeTrainableParameters(&backendGraph, model->getOptimizer());

                // Set number of output-gradients based an connection in the
                // backend-graph
                setNumberOfOutputGradients(&backendGraph);

                // Compile each backend-layer for forward usage and set pointers
                // of input and output memory
                index_t inputIdx = model->inputs_.front()->getGlobalIndex();
                std::shared_ptr<dnnl::memory> outputMemory = nullptr;
                backendGraph.visit(
                    // Start node for traversal
                    inputIdx,
                    // visitor for the current node in the traversal
                    [&outputMemory](auto node) {
                        node->compile(PropagationKind::Forward);
                        outputMemory = node->getOutputMemory();
                        assert(outputMemory != nullptr
                               && "Output memory is null during graph-traversal");
                    },
                    // visitor for the current and the next node in the traversal
                    [&outputMemory]([[maybe_unused]] auto node, auto nextNode) {
                        nextNode->setNextInputMemory(outputMemory);
                    });

                // Compile each backend-layer for backward usage and set pointers
                // of output-gradient and input-gradient memory
                index_t outputIdx = *model->backendOutputs_.begin();

                std::vector<std::shared_ptr<dnnl::memory>> inputGradientMemory;
                index_t inputGradientCounter = 0;
                backendGraph.visitBackward(
                    // Start node for traversal
                    outputIdx,
                    // visitor for the current node in the traversal
                    [&inputGradientMemory, &inputGradientCounter](auto node) {
                        // compile the current layer
                        node->compile(PropagationKind::Backward);

                        inputGradientMemory.clear();
                        inputGradientCounter = 0;
                        for (std::size_t i = 0; i < asIndex(node->getNumberOfInputs()); ++i) {
                            inputGradientMemory.push_back(node->getInputGradientMemory(i));
                        }
                    },
                    // visitor for the current and the next node in the traversal
                    [&inputGradientMemory, &inputGradientCounter]([[maybe_unused]] auto node,
                                                                  auto prevNode) {
                        // Get input gradient memory for all
                        assert(inputGradientMemory[inputGradientCounter] != nullptr
                               && "Input-gradient memory is null during backward graph-traversal");
                        prevNode->setNextOutputGradientMemory(
                            inputGradientMemory[inputGradientCounter]);
                        ++inputGradientCounter;
                    },
                    []([[maybe_unused]] auto node) { return false; });
            }

            static DataContainer<data_t> predict(Model<data_t, MlBackend::Dnnl>* model,
                                                 const DataContainer<data_t>& x)
            {
                auto& backendGraph = model->getBackendGraph();
                index_t inputIdx = model->inputs_.front()->getGlobalIndex();
                auto inputLayer = getCheckedLayerPtr(backendGraph.getNodes().at(inputIdx));

                // Create Dnnl execution-stream
                dnnl::stream executionStream(*inputLayer->getEngine());

                // Set model input
                inputLayer->setInput(x);

                // Keep track of all node we already handled
                std::vector<bool> forwardPropagationList(backendGraph.getNumberOfNodes(), false);

                backendGraph.visitWithIndex(
                    // Start node for traversal
                    inputIdx,
                    // visitor for the current node in the traversal
                    [&executionStream, &forwardPropagationList](auto node, index_t s) {
                        node->forwardPropagate(executionStream);
                        assert(forwardPropagationList[asIndex(s)] == false);
                        forwardPropagationList[asIndex(s)] = true;
                    },
                    // visitor for the current and the next node in the traversal
                    []([[maybe_unused]] auto node, [[maybe_unused]] index_t s,
                       [[maybe_unused]] auto nextNode, [[maybe_unused]] index_t nextS) {},
                    // We cut a traversal path if the current node can merge and
                    // if we haven't yet handeld all of its predecessors
                    [&backendGraph, &forwardPropagationList]([[maybe_unused]] auto node,
                                                             index_t s) {
                        for (const auto& inEdge : backendGraph.getIncomingEdges(s)) {
                            if (!forwardPropagationList[asIndex(inEdge.begin()->getIndex())])
                                return true;
                        }
                        return false;
                    });

                forwardPropagationList =
                    std::vector<bool>(asIndex(backendGraph.getNumberOfNodes()), false);

                index_t outputIdx = *std::begin(model->backendOutputs_);
                auto outputLayer = getCheckedLayerPtr(backendGraph.getNodes().at(outputIdx));
                return outputLayer->getOutput();
            }

            static typename Model<data_t, MlBackend::Dnnl>::History
                fit(Model<data_t, MlBackend::Dnnl>* model,
                    const std::vector<DataContainer<data_t>>& x,
                    const std::vector<DataContainer<data_t>>& y, index_t epochs)
            {
                typename Model<data_t, MlBackend::Dnnl>::History trainingHistory;

                auto& backendGraph = model->getBackendGraph();
                index_t inputIdx = model->inputs_.front()->getGlobalIndex();
                auto inputLayer = getCheckedLayerPtr(backendGraph.getNodes().at(inputIdx));
                index_t outputIdx = *std::begin(model->backendOutputs_);
                auto outputLayer = getCheckedLayerPtr(backendGraph.getNodes().at(outputIdx));
                auto lossFunc = model->getLoss();

                // Create a Dnnl execution-stream that will be used during
                // forward-propagation
                dnnl::stream executionStream(*inputLayer->getEngine());

                // For all epochs
                double epochLoss = 0.0;
                double epochAccuracy = 0.0;
                index_t correct = 0.0;
                for (index_t epoch = 0; epoch < epochs; ++epoch) {
                    std::cout << "Epoch " << epoch + 1 << "/" << epochs << "\n";
                    epochLoss = 0.0;
                    epochAccuracy = 0.0;
                    correct = 0.0;
                    ProgressBar progBar(static_cast<uint32_t>(x.size()), 36);
                    for (std::size_t idx = 0; idx < x.size(); ++idx) {
                        // Set this model's input as the input of the model's
                        // input layer.
                        inputLayer->setInput(x[asIndex(idx)]);

                        // Keep track of all nodes we already handled
                        std::vector<bool> nodeList(backendGraph.getNumberOfNodes(), false);

                        backendGraph.visitWithIndex(
                            // Start node for traversal
                            inputIdx,
                            // visitor for the current node in the traversal
                            [&executionStream, &nodeList](auto node, index_t s) {
                                node->forwardPropagate(executionStream);
                                assert(nodeList[asIndex(s)] == false);
                                nodeList[asIndex(s)] = true;
                            },
                            // visitor for the current and the next node in the traversal
                            []([[maybe_unused]] auto node, [[maybe_unused]] index_t s,
                               [[maybe_unused]] auto nextNode, [[maybe_unused]] index_t nextS) {},
                            // We cut a traversal path if we haven't handled all
                            // predecessors of the current node
                            [&backendGraph, &nodeList]([[maybe_unused]] auto node, index_t s) {
                                for (const auto& inEdge : backendGraph.getIncomingEdges(s)) {
                                    if (!nodeList[asIndex(inEdge.begin()->getIndex())])
                                        return true;
                                }
                                return false;
                            });

                        nodeList = std::vector<bool>(backendGraph.getNumberOfNodes(), false);

                        auto output = outputLayer->getOutput();

                        // Get accuracy
                        auto label = Utils::Encoding::fromOneHot(output, 10);
                        for (int i = 0; i < model->batchSize_; ++i) {
                            if (label[i] == y[asIndex(idx)][i]) {
                                correct += 1;
                            }
                        }

                        epochAccuracy = (static_cast<double>(correct)
                                         / static_cast<double>(((idx + 1) * model->batchSize_)));

                        // Loss calculation
                        trainingHistory.loss.push_back(lossFunc(output, y[asIndex(idx)]));
                        epochLoss += trainingHistory.loss.back();

                        ++progBar;
                        std::string preMessage =
                            std::to_string(idx) + "/" + std::to_string(x.size()) + " ";
                        progBar.display(
                            preMessage,
                            "- " + lossFunc.getName() + ": "
                                + std::to_string(epochLoss / static_cast<double>(idx + 1))
                                + " - Accuracy: " + std::to_string(epochAccuracy));

                        outputLayer->setOutputGradient(
                            lossFunc.getLossGradient(output, y[asIndex(idx)]));

                        // Backward-propagate all nodes, starting at the output
                        // until we reach the input.
                        backendGraph.visitBackwardWithIndex(
                            // Start node for traversal
                            outputIdx,
                            // Visitor for the current node in the traversal.
                            [&executionStream, &nodeList](auto node, index_t s) {
                                node->backwardPropagate(executionStream);
                                assert(nodeList[asIndex(s)] == false);
                                nodeList[asIndex(s)] = true;
                            },
                            // Visitor for the current and the next node in the
                            // traversal.
                            []([[maybe_unused]] auto node, [[maybe_unused]] index_t s,
                               [[maybe_unused]] auto nextNode, [[maybe_unused]] index_t nextS) {},
                            [&backendGraph, &nodeList]([[maybe_unused]] auto node, index_t s) {
                                for (const auto& outEdge : backendGraph.getOutgoingEdges(s)) {
                                    if (!nodeList[asIndex(outEdge.end()->getIndex())])
                                        return true;
                                }
                                return false;
                            });

                        // Set a barrier for the backward-stream.
                        executionStream.wait();

                        // If we are done with this batch, we update all
                        // trainable parameters. This also resets all
                        // accumulated gradients.
                        for (auto&& node : backendGraph.getNodes()) {
                            auto layer = getCheckedLayerPtr(node.second);
                            if (layer->isTrainable()) {
                                std::static_pointer_cast<DnnlTrainableLayer<data_t>>(layer)
                                    ->updateTrainableParameters();
                            }
                        }
                    }
                    progBar.done(std::to_string(x.size()) + "/" + std::to_string(x.size()) + " ",
                                 "");
                }
                return trainingHistory;
            }
        };
    } // namespace detail
} // namespace elsa::ml
