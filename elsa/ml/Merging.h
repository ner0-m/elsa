#pragma once

#include <string>

#include "elsaDefines.h"
#include "Layer.h"
#include "Common.h"

namespace elsa::ml
{
    template <typename data_t = real_t>
    class Merging : public Layer<data_t>
    {
    public:
        Merging(LayerType layerType, std::initializer_list<Layer<data_t>*> inputs,
                const std::string& name = "");

        bool canMerge() const override;
    };

    /// Sum layer
    ///
    /// This layer takes a list of inputs, all of the same shape, and returns
    /// a single output, also of the same shape, which is the sum of all inputs.
    template <typename data_t = real_t>
    class Sum : public Merging<data_t>
    {
    public:
        /// Construct a sum layer by specifying a list of input
        Sum(std::initializer_list<Layer<data_t>*> inputs, const std::string& name = "");

        /// \copydoc Layer::computeOutputDescriptor
        void computeOutputDescriptor() override;
    };

    /// Layer that concatenates a list of inputs.
    ///
    /// The input of this layer is a list of DataContainers, all of the same
    /// same, except for the concatenation axis. It outputs a single
    //  DataContainer that is the concatenation of all inputs
    template <typename data_t = real_t>
    class Concatenate : public Merging<data_t>
    {
    public:
        /// Construct a Concatenate layer.
        Concatenate(index_t axis, std::initializer_list<Layer<data_t>*> inputs,
                    const std::string& name = "");

        /// \copydoc Layer::computeOutputDescriptor
        void computeOutputDescriptor() override;

    private:
        index_t axis_;
    };

} // namespace elsa::ml