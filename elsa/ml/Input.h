#pragma once

#include <string>

#include "elsaDefines.h"
#include "Layer.h"

namespace elsa::ml
{
    /// Input layer
    ///
    /// \author David Tellenbach
    ///
    /// A layer representing a network's input.
    template <typename data_t = real_t>
    class Input : public Layer<data_t>
    {
    public:
        /// Construct an input layer
        ///
        /// \param inputDescriptor Descriptor for the layer's input
        /// \param batchSize Batch size
        /// \param name The layer's name. This parameter is optional and defaults to "none"
        Input(const VolumeDescriptor& inputDescriptor, index_t batchSize = 1,
              const std::string& name = "");

        /// \returns the batch size
        index_t getBatchSize() const;

    private:
        index_t batchSize_;
    };

} // namespace elsa::ml
