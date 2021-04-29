#pragma once

#include <string>
#include <vector>
#include <memory>
#include <ostream>
#include <iomanip>

#include "elsaDefines.h"
#include "VolumeDescriptor.h"
#include "Common.h"
#include "State.h"
#include "DataContainer.h"

namespace elsa::ml
{
    /// Base class for all layers in a network
    ///
    /// \author David Tellenbach
    template <typename data_t = real_t>
    class Layer
    {
    public:
        Layer() = default;

        /**
         *  Construct a layer
         *
         * \param layerType the type of the layer
         * \param requiredNumberOfDimensions the required number of
         *            dimensions any input to this layer has. This can be set
         *            to Layer::AnyNumberOfInputDimensions to indicate that the
         *            number of input dimensions is not restricted.
         *  \param allowedNumberOfInputs the allowed number of inputs to
         *             this layer. This can be set to Layer::AnyNumberOfInputs
         *             to indicate that the layer accepts an arbitrary number of
         *             inputs.
         */
        Layer(LayerType layerType, const std::string& name,
              int requiredNumberOfDimensions = Layer::AnyNumberOfInputDimensions,
              int allowedNumberOfInputs = Layer::AnyNumberOfInputs, bool isTrainable = false);

        /// Copy constructor
        Layer(const Layer&);

        /// Move constructor
        Layer(Layer&&) = default;

        /// Copy-assignment operator
        Layer& operator=(const Layer&);

        /// Move assignment operator
        Layer& operator=(Layer&&) = default;

        /// Destructor
        virtual ~Layer() = default;

        /// \returns this layer's layer-type
        LayerType getLayerType() const;

        /// \returns this layer's name
        std::string getName() const;

        /// set the layer's input descriptor at a given index
        void setInputDescriptor(const VolumeDescriptor&);

        /// \returns this layer's input descriptor at a given index
        VolumeDescriptor getInputDescriptor(index_t index = 0) const;

        /// \returns the number of inputs of this layer
        index_t getNumberOfInputs() const;

        /// \returns the layer's output descriptor
        VolumeDescriptor getOutputDescriptor() const;

        /// \returns the layer's unique global index
        index_t getGlobalIndex() const;

        /// Set this layer's input.
        void setInput(Layer* layer);

        /// Set his layer's input.
        void setInput(std::initializer_list<Layer*>);

        /// Compute this layer's output descriptor.
        virtual void computeOutputDescriptor();

        /// \returns true of a layer is trainable and false otherwise
        bool isTrainable() const;

        /// \returns true of a layer can merge multiple inputs and false otherwise
        virtual bool canMerge() const;

        /// \returns the number of trainable parameters
        index_t getNumberOfTrainableParameters() const;

        template <typename T>
        friend std::ostream& operator<<(std::ostream& os, const Layer<T>& layer);

    protected:
        /// check if the number of dimensions of the layer's input descriptor matches the
        /// required number of dimensions
        void checkNumberOfInputDimensions(const VolumeDescriptor&) const;

        static constexpr int AnyNumberOfInputDimensions = -1;
        static constexpr int AnyNumberOfInputs = -1;

        int requiredNumberOfDimensions_;
        int allowedNumberOfInputs_;

        index_t globalIndex_;
        LayerType layerType_;

        std::string name_;

        index_t numberOfTrainableParameters_;

        // A layer can have more than one input (e.g. any merging layer) but has
        // always a single well-defined output descriptor that in general depend
        // on the input descriptors
        std::vector<std::unique_ptr<DataDescriptor>> inputDescriptors_;
        std::unique_ptr<DataDescriptor> outputDescriptor_;

        bool isTrainable_;

    private:
        static index_t staticGlobalIndex_;
    };

    template <typename data_t>
    index_t Layer<data_t>::staticGlobalIndex_{0};

    template <typename T>
    std::ostream& operator<<(std::ostream& os, const Layer<T>& layer)
    {
        std::stringstream ss;

        // Name and layer-type
        std::string description =
            layer.getName() + " (" + detail::getEnumMemberAsString(layer.getLayerType()) + ")";
        os << std::left << std::setw(35) << description;

        // output-shape
        Eigen::IOFormat fmt(Eigen::StreamPrecision, 0, ", ", ", ", "", "", "(", ")");
        ss << layer.getOutputDescriptor().getNumberOfCoefficientsPerDimension().format(fmt);
        std::string outputShape = ss.str();
        os << std::left << std::setw(20) << outputShape;

        // number of parameters
        os << std::left << std::setw(10) << layer.getNumberOfTrainableParameters();

        return os;
    }

} // namespace elsa::ml