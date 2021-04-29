#include "Model.h"

#include <deque>

namespace elsa::ml
{
    template <typename data_t, MlBackend Backend>
    Model<data_t, Backend>::Model(std::initializer_list<Input<data_t>*> inputs,
                                  std::initializer_list<Layer<data_t>*> outputs,
                                  const std::string& name)
        : name_(name), inputs_(inputs), outputs_(outputs)
    {
        // Save the batch-size this model uses
        batchSize_ = inputs_.front()->getBatchSize();

        // Set all input-descriptors by traversing the graph
        setInputDescriptors();
    }

    template <typename data_t, MlBackend Backend>
    Model<data_t, Backend>::Model(Input<data_t>* input, Layer<data_t>* output,
                                  const std::string& name)
        : Model({input}, {output}, name)
    {
    }

    template <typename data_t, MlBackend Backend>
    index_t Model<data_t, Backend>::getBatchSize() const
    {
        return batchSize_;
    }

    template <typename data_t, MlBackend Backend>
    void Model<data_t, Backend>::setInputDescriptors()
    {
        // TODO(tellenbach): Replace by Graph::visit method

        // Get the graph
        auto& graph = detail::State<data_t>::getGraph();

        // Get all nodes of the graph, i.e., a map with node-indices as keys
        // and nodes as values
        auto& nodes = graph.getNodes();

        // We maintain a list of nodes we've already visited and a call-queue to
        // ensure the correct order of traversal.
        std::vector<bool> visited(asIndex(graph.getNumberOfNodes()));
        // Note that this queue is in fact a deque, so we can push and pop from
        // both, the front and back.
        std::deque<index_t> queue;

        // Perform an iterative depth-first traversal through the graph
        for (auto in : inputs_) {
            // Push the input-node onto the call-queue
            queue.push_back(in->getGlobalIndex());

            while (!queue.empty()) {
                // The current node is the top of the stack and we compute its
                // output descriptor
                index_t s = queue.back();
                queue.pop_back();

                if (!visited[static_cast<std::size_t>(s)]) {
                    // If the current node is a merging layer, its
                    // output-descriptor can depend on *all* input-descriptors.
                    // We therefore have to make sure that we really set all
                    // input-descriptors before attempting to compute a merging
                    // layer's output-descriptor or attempting to continue the
                    // traversal.
                    //
                    // We do this by checking if the number of edges that reach
                    // a merging layer is equal to the number of set inputs.
                    //
                    //      +---------+
                    //      | Layer 1 |   +---------+
                    //      +---------+   | Layer 2 |
                    //           |        +---------+
                    //           v             |
                    //      +---------+        |
                    //      | Merging |<-------+
                    //      +---------+
                    //  (?) Do we have the input from Layer1 *and* Layer2?
                    //
                    // We also have to make sure that a merging layer get's
                    // visited again when all of its inputs are set. Pushing the
                    // layer on top of the queue again causes an infinite loop
                    // since we will always visit it again, see that we can't
                    // compute its output-descriptor yet, visit it again...
                    //
                    // To solve this problem we push a merging layer to the
                    // *front* of the queue such that it get's visited again,
                    // in a delayed fashion.
                    if (!nodes.at(s).getData()->canMerge()
                        || nodes.at(s).getData()->getNumberOfInputs()
                               == static_cast<index_t>(graph.getIncomingEdges(s).size())) {
                        // We end up here if we either have no merging layer
                        // or if we have a merging layer but already gathered
                        // all of its inputs
                        nodes.at(s).getData()->computeOutputDescriptor();

                        visited[asIndex(s)] = true;
                    } else {
                        // We end up here if we have a merging layer but haven't
                        // collected all of its inputs yet. In this case we
                        // push the layer to the *front* of out call-queue.
                        queue.push_front(s);

                        // Make sure we don't handle a merging layer's childs
                        // before handling the layer itself
                        continue;
                    }
                }

                // TODO(tellenbach): Stop if we reach one of the model's output
                // layers

                // Consider all outgoing edges of a node and set their
                // input-descriptors to the output-descriptor of their parent
                // node
                for (auto& e : graph.getOutgoingEdges(s)) {
                    auto idx = e.end()->getIndex();

                    // If we haven't visited this child node yet, add it to the
                    // call-queue and set its input-descriptor
                    if (!visited[static_cast<std::size_t>(idx)]) {
                        queue.push_back(idx);
                        e.end()->getData()->setInputDescriptor(
                            nodes.at(s).getData()->getOutputDescriptor());
                    }
                }
            }
        }
    }

    template <typename data_t, MlBackend Backend>
    void Model<data_t, Backend>::compile(const Loss<data_t>& loss, Optimizer<data_t>* optimizer)
    {
        loss_ = loss;
        optimizer_ = optimizer;
        detail::BackendAdaptor<data_t, Backend>::constructBackendGraph(this);
    }

    template <typename data_t, MlBackend Backend>
    typename Model<data_t, Backend>::History
        Model<data_t, Backend>::fit(const std::vector<DataContainer<data_t>>& x,
                                    const std::vector<DataContainer<data_t>>& y, index_t epochs)
    {
        // Check if all elements of x have the same data-container
        if (std::adjacent_find(x.begin(), x.end(),
                               [](const auto& dc0, const auto& dc1) {
                                   return dc0.getDataDescriptor() != dc1.getDataDescriptor();
                               })
            != x.end())
            throw std::invalid_argument("All elements of x must have the same data-descriptor");

        // Check if all elements of y have the same data-container
        if (std::adjacent_find(y.begin(), y.end(),
                               [](const auto& dc0, const auto& dc1) {
                                   return dc0.getDataDescriptor() != dc1.getDataDescriptor();
                               })
            != y.end())
            throw std::invalid_argument("All elements of y must have the same data-descriptor");

        return detail::BackendAdaptor<data_t, Backend>::fit(this, x, y, epochs);
    }

    template <typename data_t, MlBackend Backend>
    DataContainer<data_t> Model<data_t, Backend>::predict(const DataContainer<data_t>& x)
    {
        return detail::BackendAdaptor<data_t, Backend>::predict(this, x);
    }

    template <typename data_t, MlBackend Backend>
    Optimizer<data_t>* Model<data_t, Backend>::getOptimizer()
    {
        return optimizer_;
    }

    template <typename data_t, MlBackend Backend>
    std::string Model<data_t, Backend>::getName() const
    {
        return name_;
    }

    template <typename data_t, MlBackend Backend>
    std::vector<Input<data_t>*> Model<data_t, Backend>::getInputs()
    {
        return inputs_;
    }

    template <typename data_t, MlBackend Backend>
    std::vector<Layer<data_t>*> Model<data_t, Backend>::getOutputs()
    {
        return outputs_;
    }

    template <typename data_t, MlBackend Backend>
    detail::Graph<typename detail::BackendSelector<data_t, Backend, LayerType::Undefined>::Type,
                  false>&
        Model<data_t, Backend>::getBackendGraph()
    {
        return backendGraph_;
    }

    template <typename data_t, MlBackend Backend>
    const detail::Graph<
        typename detail::BackendSelector<data_t, Backend, LayerType::Undefined>::Type, false>&
        Model<data_t, Backend>::getBackendGraph() const
    {
        return backendGraph_;
    }

    template <typename data_t, MlBackend Backend>
    const Loss<data_t>& Model<data_t, Backend>::getLoss() const
    {
        return loss_;
    }

    template class Model<float, MlBackend::Dnnl>;
    template class Model<float, MlBackend::Cudnn>;
} // namespace elsa::ml