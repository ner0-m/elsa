#include <sstream>
#include <algorithm>
#include "TypeCasts.hpp"

#include "Layer.h"

namespace elsa::ml
{
    template <typename data_t>
    Layer<data_t>::Layer(LayerType layerType, const std::string& name,
                         int requiredNumberOfDimensions, int allowedNumberOfInputs,
                         bool isTrainable)
        : requiredNumberOfDimensions_(requiredNumberOfDimensions),
          allowedNumberOfInputs_(allowedNumberOfInputs),
          layerType_(layerType),
          numberOfTrainableParameters_(0),
          isTrainable_(isTrainable)
    {
        globalIndex_ = staticGlobalIndex_++;

        // If no name is set we use a (lowercase) string representation of the
        // layer-type together with the layer's global index
        if (name == "") {
            std::stringstream ss;
            std::string typeString = detail::getEnumMemberAsString(layerType);
            std::transform(std::begin(typeString), std::end(typeString), std::begin(typeString),
                           [](unsigned char c) { return std::tolower(c); });
            ss << typeString << "_" << globalIndex_;
            name_ = ss.str();
        } else {
            name_ = name;
        }

        if (this->isTrainable())
            staticGlobalIndex_++;
    }

    template <typename data_t>
    Layer<data_t>::Layer(const Layer& other)
        : globalIndex_(other.globalIndex_),
          layerType_(other.layerType_),
          name_(other.name_),
          outputDescriptor_(other.outputDescriptor_->clone())
    {
        for (std::size_t i = 0; i < other.inputDescriptors_.size(); ++i)
            inputDescriptors_.push_back(other.inputDescriptors_[i]->clone());
    }

    template <typename data_t>
    Layer<data_t>& Layer<data_t>::operator=(const Layer<data_t>& other)
    {
        if (this != &other) {
            globalIndex_ = other.globalIndex_;
            layerType_ = other.layerType_;
            name_ = other.name_;
            outputDescriptor_ = other.outputDescriptor_->clone();
            for (std::size_t i = 0; i < other.inputDescriptors_.size(); ++i)
                inputDescriptors_.push_back(other.inputDescriptors_[i]->clone());
        }
        return *this;
    }

    template <typename data_t>
    LayerType Layer<data_t>::getLayerType() const
    {
        return layerType_;
    }

    template <typename data_t>
    std::string Layer<data_t>::getName() const
    {
        return name_;
    }

    template <typename data_t>
    void Layer<data_t>::setInputDescriptor(const VolumeDescriptor& inputDescriptor)
    {
        Layer<data_t>::checkNumberOfInputDimensions(inputDescriptor);
        inputDescriptors_.push_back(inputDescriptor.clone());
    }

    template <typename data_t>
    void Layer<data_t>::setInput(Layer<data_t>* layer)
    {
        // Get graph from global state
        auto& graph = detail::State<data_t>::getGraph();
        // Add edge from layer to this and fill the graph nodes
        graph.addEdge(layer->getGlobalIndex(), globalIndex_);
        graph.setData(layer->getGlobalIndex(), layer);
        graph.setData(globalIndex_, this);

        // Note that we don't set input descriptors here since we want to allow delaying the
        // input specifications. We'll set all input descriptors later when building the model
        // by traversing the graph
    }

    template <typename data_t>
    void Layer<data_t>::setInput(std::initializer_list<Layer*> layers)
    {
        for (auto&& l : layers)
            setInput(l);
    }

    template <typename data_t>
    VolumeDescriptor Layer<data_t>::getInputDescriptor(index_t index) const
    {
        if (inputDescriptors_.size() <= static_cast<std::size_t>(index))
            throw std::logic_error("No input descriptor at given index");

        if (!inputDescriptors_[asUnsigned(index)])
            throw std::logic_error("Input descriptor not set");

        return *detail::dynamic_unique_ptr_cast<VolumeDescriptor>(
            inputDescriptors_[asUnsigned(index)]->clone());
    }

    template <typename data_t>
    index_t Layer<data_t>::getNumberOfInputs() const
    {
        return static_cast<index_t>(inputDescriptors_.size());
    }

    template <typename data_t>
    VolumeDescriptor Layer<data_t>::getOutputDescriptor() const
    {
        if (!outputDescriptor_)
            throw std::logic_error("Output descriptor not set");

        return *detail::dynamic_unique_ptr_cast<VolumeDescriptor>(outputDescriptor_->clone());
    }

    template <typename data_t>
    index_t Layer<data_t>::getGlobalIndex() const
    {
        return globalIndex_;
    }

    template <typename data_t>
    void Layer<data_t>::computeOutputDescriptor()
    {
        // This default implementation requires a single input descriptor
        if (inputDescriptors_.size() != 1 || !inputDescriptors_.front())
            throw std::logic_error("Cannot compute output descriptor since it depends on an input "
                                   "descriptor that has not been set");
        outputDescriptor_ = inputDescriptors_.front()->clone();
    }

    template <typename data_t>
    void Layer<data_t>::checkNumberOfInputDimensions(const VolumeDescriptor& inputDescriptor) const
    {
        if (requiredNumberOfDimensions_ != Layer<data_t>::AnyNumberOfInputDimensions
            && requiredNumberOfDimensions_ != inputDescriptor.getNumberOfDimensions()) {
            std::stringstream what;
            what << "Expected an input descriptor with " << requiredNumberOfDimensions_
                 << " dimension but got one with " << inputDescriptor.getNumberOfDimensions()
                 << ".";
            throw std::invalid_argument(what.str());
        }
    }

    template <typename data_t>
    bool Layer<data_t>::isTrainable() const
    {
        return isTrainable_;
    }

    template <typename data_t>
    bool Layer<data_t>::canMerge() const
    {
        return false;
    }

    template <typename data_t>
    index_t Layer<data_t>::getNumberOfTrainableParameters() const
    {
        return numberOfTrainableParameters_;
    }

    template class Layer<float>;
} // namespace elsa::ml
