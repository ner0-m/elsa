#pragma once

#include "DnnlTrainableLayer.h"
#include "DataDescriptor.h"
#include "DataContainer.h"
#include "RandomInitializer.h"

#include "dnnl.hpp"

namespace elsa
{
    template <typename data_t>
    class DnnlConv final : public DnnlTrainableLayer<data_t>
    {
    public:
        using BaseType = DnnlTrainableLayer<data_t>;

        DnnlConv(const DataDescriptor& inputDescriptor, const DataDescriptor& outputDescriptor,
                 const DataDescriptor& weightsDescriptor, const IndexVector_t& strideVector,
                 const IndexVector_t& paddingVector, Initializer initializer);

        void compile() override;

    private:
        using BaseType::_weightsDimensions;
        using BaseType::_weightsMemoryDescriptor;
        using BaseType::_weightsMemory;
        using BaseType::_reorderedWeightsMemory;
        using BaseType::_biasDimensions;
        using BaseType::_biasMemoryDescriptor;
        using BaseType::_biasMemory;
        using BaseType::_srcMemoryDescriptor;
        using BaseType::_reorderedSrcMemory;
        using BaseType::_dstMemoryDescriptor;
        using BaseType::_engine;
        using BaseType::_forwardPrimitives;
        using BaseType::_dstMemory;
        using BaseType::_srcMemory;
        using BaseType::_forwardArguments;
        using BaseType::_typeTag;
        using BaseType::_hasReorderedMemory;

        Initializer _initializer = Initializer::Uniform;

        dnnl::memory::dims _paddingDimensions;

        dnnl::memory::dims _strideDimensions;

        dnnl::convolution_forward::primitive_desc _forwardPrimitiveDescriptor;
    };
} // namespace elsa