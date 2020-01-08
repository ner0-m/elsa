#include "elsa.h"

static auto logger = Logger::get("MnistExample");

auto constructAlexNet()
{
    IndexVector_t inputDims(3);
    // Batch: 10, Height and width: 28
    inputDims << 10, 28, 28;
    DataDescriptor inputDesc(inputDims);

    auto model = SequentialNetwork<float, MlBackend::Dnnl>(inputDesc);

    model
        // Conv layer with 6 filters of size 5x5
        .addConvLayer({6, 5, 5})
        // Elu activation layer
        .addActivationLayer(Activation::Elu)
        // Conv layer with 6 filters of size 5x5
        .addConvLayer({6, 5, 5})
        // Elu activation layer
        .addActivationLayer(Activation::Elu)
        // Max pooling layer with pooling window 2x2
        .addPoolingLayer({2, 2})
        // Dense layer with 128 neurons
        .addDenseLayer(128)
        // Dense layer with 10 neurons
        .addDenseLayer(10)
        // Softmax layer
        .addSoftmaxLayer();

    model.setLearningRate(0.03);

    model.compile();

    auto finalError = model.train(data, labels);

    logger->info("Finished training with final error {}", finalError);
}