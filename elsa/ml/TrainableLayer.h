#pragma once

#include "DataDescriptor.h"
#include "Layer.h"
#include "elsaDefines.h"
#include "RandomInitializer.h"

namespace elsa
{
    template <typename data_t, MlBackend Backend>
    class TrainableLayer : public Layer<data_t, Backend>
    {
    public:
        static Initializer initializer;

        const DataDescriptor& getWeightsDescriptor() const;
        const DataDescriptor& getBiasDescriptor() const;

    protected:
        using BaseType = Layer<data_t, Backend>;

        TrainableLayer(const DataDescriptor& inputDescriptor,
                       const DataDescriptor& weightsDescriptor);

        /// \copydoc Layer::_inputDescriptor
        using BaseType::_inputDescriptor;

        /// \copydoc Layer::_outputDescriptor
        using BaseType::_outputDescriptor;

        /// \copydoc Layer::_backend
        using BaseType::_backend;

        /// \copydoc Layer::_weightsDescriptor
        DataDescriptor _weightsDescriptor;

        /// \copydoc Layer::_biasDescriptor
        DataDescriptor _biasDescriptor;
    };

    template <typename data_t, MlBackend Backend>
    Initializer TrainableLayer<data_t, Backend>::initializer = Initializer::Uniform;

    template <typename data_t, MlBackend Backend>
    inline TrainableLayer<data_t, Backend>::TrainableLayer(const DataDescriptor& inputDescriptor,
                                                           const DataDescriptor& weightsDescriptor)
        : Layer<data_t, Backend>(inputDescriptor), _weightsDescriptor(weightsDescriptor)
    {
        IndexVector_t biasDims(1);
        biasDims << _weightsDescriptor.getNumberOfCoefficientsPerDimension()[0];
        _biasDescriptor = DataDescriptor(biasDims);
    }

    template <typename data_t, MlBackend Backend>
    inline const DataDescriptor& TrainableLayer<data_t, Backend>::getWeightsDescriptor() const
    {
        return _weightsDescriptor;
    }

    template <typename data_t, MlBackend Backend>
    inline const DataDescriptor& TrainableLayer<data_t, Backend>::getBiasDescriptor() const
    {
        return _biasDescriptor;
    }

} // namespace elsa