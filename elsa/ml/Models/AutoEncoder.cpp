#include "AutoEncoder.h"

namespace elsa::ml
{
    template <typename data_t, MlBackend Backend>
    AutoEncoder<data_t, Backend>::AutoEncoder(VolumeDescriptor inputDescriptor, index_t batchSize)
        : Model<data_t, Backend>(),
          _input(ml::Input<data_t>(inputDescriptor, batchSize)),
          _conv1x1(ml::Conv2D<data_t>(inputDescriptor.getNumberOfCoefficientsPerDimension()[2],
                                      {1, 1, 16}, ml::Activation::Relu, 1, ml::Padding::Same))
    {
        name_ = "AutoEncoder";

        _convContr1.setInput(&_input);
        _maxPool1.setInput(&_convContr1);

        _convContr2.setInput(&_maxPool1);
        _maxPool2.setInput(&_convContr2);

        _convContr3.setInput(&_maxPool2);

        _upsample1.setInput(&_convContr3);
        _convExpan1.setInput(&_upsample1);

        _upsample2.setInput(&_convExpan1);
        _convExpan2.setInput(&_upsample2);

        _conv1x1.setInput(&_convExpan2);

        inputs_ = {&_input};
        outputs_ = {&_conv1x1};

        // save the batch-size this model uses
        batchSize_ = inputs_.front()->getBatchSize();

        // set all input-descriptors by traversing the graph
        setInputDescriptors();
    }
    // TODO add here more code

    template class AutoEncoder<float, MlBackend::Dnnl>;
    template class AutoEncoder<float, MlBackend::Cudnn>;
} // namespace elsa::ml