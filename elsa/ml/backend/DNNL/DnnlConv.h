#pragma once

#include "DnnlLayer.h"
#include "DataDescriptor.h"
#include "DataContainer.h"
#include "RandomInitializer.h"

#include "dnnl.hpp"

namespace elsa
{
    template <typename data_t>
    class DnnlConv final : public DnnlLayer<data_t>
    {
    public:
        using BaseType = DnnlLayer<data_t>;

        DnnlConv(const DataDescriptor& inputDescriptor, const DataDescriptor& outputDescriptor,
                 const DataDescriptor& weightsDescriptor, const IndexVector_t& strideVector,
                 const IndexVector_t& paddingVector);

        void setWeights(const DataContainer<data_t>& weights);
        void setBias(const DataContainer<data_t>& bias);
        void setInitializer(Initializer initializer);

        virtual void compile() override;

    private:
        Initializer _initializer = Initializer::Uniform;

        dnnl::memory _reorderedSrcMemory;

        /// The dimension of the convolutional layer's weights
        dnnl::memory::dims _weightsDimensions;
        dnnl::memory::desc _weightsMemoryDescriptor;
        dnnl::memory _weightsMemory;
        dnnl::memory _reorderedWeightsMemory;

        dnnl::memory::dims _biasDimensions;
        dnnl::memory::desc _biasMemoryDescriptor;
        dnnl::memory _biasMemory;

        dnnl::memory::dims _paddingDimensions;

        dnnl::memory::dims _strideDimensions;

        dnnl::convolution_forward::primitive_desc _forwardPrimitiveDescriptor;

        using BaseType::_srcMemoryDescriptor;
        using BaseType::_dstMemoryDescriptor;
        using BaseType::_engine;
        using BaseType::_forwardPrimitives;
        using BaseType::_dstMemory;
        using BaseType::_srcMemory;
        using BaseType::_forwardArguments;
        using BaseType::_typeTag;
        using BaseType::_hasReorderedMemory;
    };
} // namespace elsa