#pragma once

#include <vector>
#include <string>

#include "Common.h"
#include "Layer.h"
#include "Conv.h"

namespace elsa::ml
{
    template <typename data_t>
    class Pooling : public Layer<data_t>
    {
    public:
        IndexVector_t getPoolSize() const;
        index_t getStrides() const;
        Padding getPadding() const;

    protected:
        Pooling(LayerType layerType, index_t poolingDimensions, const IndexVector_t& poolSize,
                index_t strides, Padding padding, const std::string& name);

    private:
        void computeOutputDescriptor() override;

        index_t poolingDimensions_;
        IndexVector_t poolSize_;
        index_t strides_;
        Padding padding_;
    };

    template <typename data_t = real_t>
    struct MaxPooling1D : public Pooling<data_t> {
        MaxPooling1D(index_t poolSize, index_t strides = 1, Padding padding = Padding::Valid,
                     const std::string& name = "");
    };

    template <typename data_t = real_t>
    struct MaxPooling2D : public Pooling<data_t> {
        MaxPooling2D(const IndexVector_t& poolSize = IndexVector_t{{2, 2}}, index_t strides = 2,
                     Padding padding = Padding::Valid, const std::string& name = "");
    };
} // namespace elsa::ml
