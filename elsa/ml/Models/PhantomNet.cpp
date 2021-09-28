#include "PhantomNet.h"

namespace elsa::ml
{
    template <typename data_t, MlBackend Backend>
    PhantomNet<data_t, Backend>::PhantomNet(VolumeDescriptor inputDescriptor, index_t batchSize)
        : Model<data_t, Backend>(),
          _input(ml::Input<data_t>(inputDescriptor, batchSize)),
          _convStart(ml::Conv2D<data_t>(
              64, {3, 3, inputDescriptor.getNumberOfCoefficientsPerDimension()[2]},
              ml::Activation::Relu, 1, ml::Padding::Same)),
          _convEnd(ml::Conv2D<data_t>(inputDescriptor.getNumberOfCoefficientsPerDimension()[2],
                                      {3, 3, 64}, ml::Activation::Relu, 1, ml::Padding::Same))
    {
        name_ = "PhantomNet";

        _convStart.setInput(&_input);
        _tdb1.setInput(&_convStart);
        _td1.setInput(&_tdb1);

        _tdb2.setInput(&_td1);
        _td2.setInput(&_tdb2);

        _tdb3.setInput(&_td2);
        _td3.setInput(&_tdb3);

        _tdb4.setInput(&_td3);

        _tu1.setInput(&_tdb4);
        _tdb5.setInput(&_tu1);

        _tu2.setInput(&_tdb5);
        _tdb6.setInput(&_tu2);

        _tu3.setInput(&_tdb6);
        _tdb7.setInput(&_tu3);

        _convEnd.setInput(&_tdb7);

        inputs_ = {&_input};
        outputs_ = {&_convEnd};

        // save the batch-size this model uses
        batchSize_ = inputs_.front()->getBatchSize();

        // set all input-descriptors by traversing the graph
        setInputDescriptors();
    }

    template <typename data_t, MlBackend Backend>
    PhantomNet<data_t, Backend>::TrimmedDenseBlock4::TrimmedDenseBlock4(index_t in_channels,
                                                                        index_t growth_rate)
        : Layer<data_t>(LayerType::Undefined, "TrimmedDenseBlock4"),
          _conv1{ml::Conv2D<data_t>(growth_rate, {3, 3, in_channels}, ml::Activation::Relu, 1,
                                    ml::Padding::Same)},
          _conv2{ml::Conv2D<data_t>(growth_rate, {3, 3, 1 * growth_rate + in_channels},
                                    ml::Activation::Relu, 1, ml::Padding::Same)},
          _conv3{ml::Conv2D<data_t>(growth_rate, {3, 3, 2 * growth_rate + in_channels},
                                    ml::Activation::Relu, 1, ml::Padding::Same)},
          _conv4{ml::Conv2D<data_t>(growth_rate, {3, 3, 3 * growth_rate + in_channels},
                                    ml::Activation::Relu, 1, ml::Padding::Same)}
    {
        index_t lastAxis = _input.getInputDescriptor().getNumberOfDimensions() - 1;

        inputs_ = {&_input};

        ml::Tanh tanh1;
        _conv1.setInput(&_input);
        tanh1.setInput(&_conv1);

        ml::Tanh tanh2;
        _conv2.setInput(ml::Concatenate(lastAxis, {&_input, &tanh1}));
        tanh2.setInput(&_conv2);

        ml::Tanh tanh3;
        _conv3.setInput(ml::Concatenate(lastAxis, {&_input, &tanh1, &tanh2}));
        tanh3.setInput(&_conv3);

        ml::Tanh tanh4;
        _conv4.setInput(ml::Concatenate(lastAxis, {&_input, &tanh1, &tanh2, &tanh3}));
        tanh4.setInput(&_conv4);

        ml::Concatenate lastConc(lastAxis, {&tanh1, &tanh2, &tanh3, &tanh4});

        outputs_ = {&lastConc};

        // save the batch-size this model uses
        batchSize_ = inputs_.front()->getBatchSize();

        // set all input-descriptors by traversing the graph
        setInputDescriptors();
    }

    template <typename data_t, MlBackend Backend>
    PhantomNet<data_t, Backend>::TrimmedDenseBlock8::TrimmedDenseBlock8(index_t in_channels,
                                                                        index_t growth_rate)

        : Layer<data_t>(LayerType::Undefined, "TrimmedDenseBlock8"),
          _conv1{ml::Conv2D<data_t>(growth_rate, {3, 3, in_channels}, ml::Activation::Relu, 1,
                                    ml::Padding::Same)},
          _conv2{ml::Conv2D<data_t>(growth_rate, {3, 3, 1 * growth_rate + in_channels},
                                    ml::Activation::Relu, 1, ml::Padding::Same)},
          _conv3{ml::Conv2D<data_t>(growth_rate, {3, 3, 2 * growth_rate + in_channels},
                                    ml::Activation::Relu, 1, ml::Padding::Same)},
          _conv4{ml::Conv2D<data_t>(growth_rate, {3, 3, 3 * growth_rate + in_channels},
                                    ml::Activation::Relu, 1, ml::Padding::Same)},
          _conv5{ml::Conv2D<data_t>(growth_rate, {3, 3, 4 * growth_rate + in_channels},
                                    ml::Activation::Relu, 1, ml::Padding::Same)},
          _conv6{ml::Conv2D<data_t>(growth_rate, {3, 3, 5 * growth_rate + in_channels},
                                    ml::Activation::Relu, 1, ml::Padding::Same)},
          _conv7{ml::Conv2D<data_t>(growth_rate, {3, 3, 6 * growth_rate + in_channels},
                                    ml::Activation::Relu, 1, ml::Padding::Same)},
          _conv8{ml::Conv2D<data_t>(growth_rate, {3, 3, 7 * growth_rate + in_channels},
                                    ml::Activation::Relu, 1, ml::Padding::Same)}
    {
        index_t lastAxis = _input.getInputDescriptor().getNumberOfDimensions() - 1;

        inputs_ = {&_input};

        ml::Tanh<data_t> tanh1;
        _conv1.setInput(&_input);
        tanh1.setInput(&_conv1);

        ml::Tanh<data_t> tanh2;
        _conv2.setInput(ml::Concatenate(lastAxis, {&_input, &tanh1}));
        tanh2.setInput(&_conv2);

        ml::Tanh<data_t> tanh3;
        _conv3.setInput(ml::Concatenate(lastAxis, {&_input, &tanh1, &tanh2}));
        tanh3.setInput(&_conv3);

        ml::Tanh<data_t> tanh4;
        _conv4.setInput(ml::Concatenate(lastAxis, {&_input, &tanh1, &tanh2, &tanh3}));
        tanh4.setInput(&_conv4);

        ml::Tanh<data_t> tanh5;
        _conv5.setInput(ml::Concatenate(lastAxis, {&_input, &tanh1, &tanh2, &tanh3, &tanh4}));
        tanh5.setInput(&_conv5);

        ml::Tanh<data_t> tanh6;
        _conv6.setInput(
            ml::Concatenate(lastAxis, {&_input, &tanh1, &tanh2, &tanh3, &tanh4, &tanh5}));
        tanh6.setInput(&_conv6);

        ml::Tanh<data_t> tanh7;
        _conv7.setInput(
            ml::Concatenate(lastAxis, {&_input, &tanh1, &tanh2, &tanh3, &tanh4, &tanh5, &tanh6}));
        tanh7.setInput(&_conv7);

        ml::Tanh<data_t> tanh8;
        _conv8.setInput(ml::Concatenate(
            lastAxis, {&_input, &tanh1, &tanh2, &tanh3, &tanh4, &tanh5, &tanh6, &tanh7}));
        tanh8.setInput(&_conv8);

        ml::Concatenate lastConc(lastAxis,
                                 {&tanh1, &tanh2, &tanh3, &tanh4, &tanh5, &tanh6, &tanh7, &tanh8});

        outputs_ = {&lastConc};

        // save the batch-size this model uses
        batchSize_ = inputs_.front()->getBatchSize();

        // set all input-descriptors by traversing the graph
        setInputDescriptors();
    }

    template <typename data_t, MlBackend Backend>
    PhantomNet<data_t, Backend>::TransitionDown::TransitionDown(index_t in_channels,
                                                                index_t out_channels)

        : Layer<data_t>(LayerType::Undefined, "TransitionDown"),
          _conv{ml::Conv2D<data_t>(out_channels, {3, 3, in_channels}, ml::Activation::Tanh, 1,
                                   ml::Padding::Same)}
    {
        inputs_ = {&_input};

        _conv.setInput(&_input);
        ml::Tanh<data_t> tanh;
        tanh.setInput(&_conv);

        _maxPool.setInput(&tanh);

        outputs_ = {&_maxPool};

        // save the batch-size this model uses
        batchSize_ = inputs_.front()->getBatchSize();

        // set all input-descriptors by traversing the graph
        setInputDescriptors();
    }

    template <typename data_t, MlBackend Backend>
    PhantomNet<data_t, Backend>::TransitionUp::TransitionUp(index_t in_channels,
                                                            index_t out_channels)
        : Layer<data_t>(LayerType::Undefined, "TransitionUp"),
          _conv{ml::Conv2D<data_t>(out_channels, {3, 3, in_channels}, ml::Activation::Relu, 1,
                                   ml::Padding::Same)}
    {
        inputs_ = {&_input};

        _conv.setInput(&_input);
        _upsample.setInput(&_conv);

        outputs_ = {&_upsample};

        // save the batch-size this model uses
        batchSize_ = inputs_.front()->getBatchSize();

        // set all input-descriptors by traversing the graph
        setInputDescriptors();
    }

    template class PhantomNet<float>;
    template class PhantomNet<double>;
} // namespace elsa::ml
