#pragma once

#include <string>

#include "Common.h"
#include "Layer.h"

namespace elsa::ml
{
    /// A Softmax layer.
    ///
    /// \author David Tellenbach
    template <typename data_t = real_t>
    class Softmax : public Layer<data_t>
    {
    public:
        /// Construct a Softmax layer.
        explicit Softmax(index_t axis = -1, const std::string& name = "");

        /// Get the Softmax axis.
        index_t getAxis() const;

        /// \copydoc Layer::computeOutputDescriptor
        void computeOutputDescriptor() override;

    private:
        index_t axis_;
    };
} // namespace elsa::ml