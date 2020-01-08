#include <catch2/catch.hpp>
#include <type_traits>
#include <random>

#include "elsaDefines.h"
#include "DataDescriptor.h"
#include "ConvLayer.h"

using namespace elsa;

TEST_CASE("ConvLayer forward", "elsa_ml")
{
    SECTION("Test 1")
    {
        // Example taken from http://cs231n.github.io/convolutional-networks/
        IndexVector_t inputVec(4);
        inputVec << 1, 3, 5, 5;
        DataDescriptor inputDesc(inputVec);

        IndexVector_t weightsVec(4);
        weightsVec << 2, 3, 3, 3;
        DataDescriptor weightsDesc(weightsVec);

        IndexVector_t stridesVec(2);
        stridesVec << 2, 2;

        IndexVector_t paddingVec(2);
        paddingVec << 1, 1;

        ConvLayer<float> conv(inputDesc, weightsDesc, stridesVec, paddingVec);

        REQUIRE(conv.getOutputDescriptor().getNumberOfDimensions() == 4);

        // Batch
        REQUIRE(conv.getOutputDescriptor().getNumberOfCoefficientsPerDimension()[0] == 1);

        // Number of output channels should be equal to number of weight channels
        REQUIRE(conv.getOutputDescriptor().getNumberOfCoefficientsPerDimension()[1] == 2);

        // Output height should be 3
        REQUIRE(conv.getOutputDescriptor().getNumberOfCoefficientsPerDimension()[2] == 3);

        // Output depth should be 3
        REQUIRE(conv.getOutputDescriptor().getNumberOfCoefficientsPerDimension()[3] == 3);

        Eigen::VectorXf vec(1 * 3 * 5 * 5);

        // clang-format off
        vec <<  // First channel
            1, 2, 1, 1, 1,
            1, 2, 2, 1, 1,
            2, 0, 2, 0, 0,
            1, 1, 2, 2, 0,
            2, 1, 1, 0, 2,
            // Second channel
            1, 0, 2, 0, 2,
            1, 0, 2, 1, 2,
            0, 1, 1, 1, 1,
            1, 0, 2, 0, 1,
            2, 2, 0, 0, 1,
            // Third channel
            2, 2, 0, 1, 2,
            2, 1, 0, 0, 2,
            1, 2, 0, 1, 0,
            2, 0, 0, 2, 2,
            2, 1, 2, 0, 0;
        // clang-format on
        DataContainer<float> input(inputDesc, vec);

        Eigen::VectorXf vec2(2 * 3 * 3 * 3);
        // clang-format off
        vec2 << // First filter
              // First channel
               0,  0, -1,
              -1,  0, -1,
               1, -1,  0,
              // Second channel
               1, -1, -1,
               1,  1,  1,
              -1,  0,  1,
              // Third channel
               0,  0, -1,
               1, -1, -1,
               0, -1,  0,
              // Second filter
              // First channel
              -1, -1, -1,
              -1, -1,  1,
               0,  0,  1,
              // Second channel
              -1,  0,  1,
              -1,  0,  1,
               1, -1,  1,
              // Third channel
               0,  0,  1,
              -1,  0,  1,
              -1,  0, -1;
        // clang-format on

        DataContainer<float> weights(weightsDesc, vec2);

        Eigen::VectorXf vec3(2);
        vec3 << 1, 0;
        IndexVector_t biasVec(1);
        biasVec << 2;
        DataDescriptor biasDesc(biasVec);
        DataContainer<float> bias(biasDesc, vec3);

        auto backend = conv.getBackend();
        backend->setInput(input);
        backend->compile();
        std::static_pointer_cast<typename ConvLayer<float>::BackendLayerType>(backend)->setWeights(
            weights);
        std::static_pointer_cast<typename ConvLayer<float>::BackendLayerType>(backend)->setBias(
            bias);

        auto engine = backend->getEngine();
        dnnl::stream s(*engine);
        backend->forwardPropagate(s);
        auto output = backend->getOutput();

        Eigen::VectorXf required(18);
        // clang-format off
        required << // First channel
                    -7,  2, -2,
                    -8,  0,  3,
                    -1, -5,  1,
                    // Second channel
                     3, -4, -4,
                    -1, -9, -8,
                     0, -8, -4;
        // clang-format on

        for (int i = 0; i < 18; ++i)
            REQUIRE(output[i] == required[i]);
    }

    SECTION("Test 2")
    {
        IndexVector_t inputVec(4);
        inputVec << 1, 1, 6, 6;
        DataDescriptor inputDesc(inputVec);

        IndexVector_t weightsVec(4);
        weightsVec << 1, 1, 3, 3;
        DataDescriptor weightsDesc(weightsVec);

        IndexVector_t stridesVec(2);
        stridesVec << 1, 1;

        IndexVector_t paddingVec(2);
        paddingVec << 0, 0;

        ConvLayer<float> conv(inputDesc, weightsDesc, stridesVec, paddingVec);

        Eigen::VectorXf vec(1 * 1 * 6 * 6);

        // clang-format off
        vec <<  // First channel
                10.f, 10.f, 10.f, 0.f, 0.f, 0.f,
                10.f, 10.f, 10.f, 0.f, 0.f, 0.f,
                10.f, 10.f, 10.f, 0.f, 0.f, 0.f,
                10.f, 10.f, 10.f, 0.f, 0.f, 0.f,
                10.f, 10.f, 10.f, 0.f, 0.f, 0.f,
                10.f, 10.f, 10.f, 0.f, 0.f, 0.f;
        // clang-format on
        DataContainer<float> input(inputDesc, vec);

        Eigen::VectorXf vec2(1 * 1 * 3 * 3);
        // clang-format off
        vec2 << 1.f, 0.f, -1.f,
                1.f, 0.f, -1.f,
                1.f, 0.f, -1.f;
        // clang-format on

        DataContainer<float> weights(weightsDesc, vec2);

        Eigen::VectorXf vec3(1);
        vec3 << 0;
        IndexVector_t biasVec(1);
        biasVec << 1;
        DataDescriptor biasDesc(biasVec);
        DataContainer<float> bias(biasDesc, vec3);

        auto backend = conv.getBackend();
        backend->setInput(input);
        backend->compile();
        std::static_pointer_cast<typename ConvLayer<float>::BackendLayerType>(backend)->setWeights(
            weights);
        std::static_pointer_cast<typename ConvLayer<float>::BackendLayerType>(backend)->setBias(
            bias);

        auto engine = backend->getEngine();
        dnnl::stream s(*engine);
        backend->forwardPropagate(s);
        s.wait();
        auto output = backend->getOutput();

        Eigen::VectorXf required(16);
        // clang-format off
        required << // First channel
                    0, 30, 30, 0,
                    0, 30, 30, 0,
                    0, 30, 30, 0,
                    0, 30, 30, 0;
        // clang-format on

        for (int i = 0; i < 16; ++i)
            REQUIRE(output[i] == required[i]);
    }
}

TEST_CASE("ConvLayer backward", "elsa_ml")
{
    SECTION("Test 1")
    {
        IndexVector_t inputVec(4);
        inputVec << 1, 1, 6, 6;
        DataDescriptor inputDesc(inputVec);

        IndexVector_t weightsVec(4);
        weightsVec << 1, 1, 3, 3;
        DataDescriptor weightsDesc(weightsVec);

        IndexVector_t stridesVec(2);
        stridesVec << 1, 1;

        IndexVector_t paddingVec(2);
        paddingVec << 0, 0;

        ConvLayer<float> conv(inputDesc, weightsDesc, stridesVec, paddingVec);

        Eigen::VectorXf vec(1 * 1 * 6 * 6);

        // clang-format off
        vec <<  // First channel
                10.f, 10.f, 10.f, 0.f, 0.f, 0.f,
                10.f, 10.f, 10.f, 0.f, 0.f, 0.f,
                10.f, 10.f, 10.f, 0.f, 0.f, 0.f,
                10.f, 10.f, 10.f, 0.f, 0.f, 0.f,
                10.f, 10.f, 10.f, 0.f, 0.f, 0.f,
                10.f, 10.f, 10.f, 0.f, 0.f, 0.f;
        // clang-format on
        DataContainer<float> input(inputDesc, vec);

        Eigen::VectorXf vec2(1 * 1 * 3 * 3);
        // clang-format off
        vec2 << 1.f, 0.f, -1.f,
                1.f, 0.f, -1.f,
                1.f, 0.f, -1.f;
        // clang-format on

        DataContainer<float> weights(weightsDesc, vec2);

        Eigen::VectorXf vec3(1);
        vec3 << 0;
        IndexVector_t biasVec(1);
        biasVec << 1;
        DataDescriptor biasDesc(biasVec);
        DataContainer<float> bias(biasDesc, vec3);

        Eigen::VectorXf outputGrad(16);
        // clang-format off
        outputGrad << // First channel
                    0, 30, 30, 0,
                    0, 30, 30, 0,
                    0, 30, 30, 0,
                    0, 30, 30, 0;
        // clang-format on

        DataContainer<float> outputGradient(conv.getOutputDescriptor(), outputGrad);

        auto backend = conv.getBackend();
        backend->setInput(input);
        backend->setOutputGradient(outputGradient);

        backend->compile(PropagationKind::Full);
        std::static_pointer_cast<typename ConvLayer<float>::BackendLayerType>(backend)->setWeights(
            weights);
        std::static_pointer_cast<typename ConvLayer<float>::BackendLayerType>(backend)->setBias(
            bias);

        auto engine = backend->getEngine();
        dnnl::stream s(*engine);
        backend->forwardPropagate(s);
        s.wait();
        auto output = backend->getOutput();
        backend->backwardPropagate(s);
        auto gradientWeights =
            std::static_pointer_cast<typename decltype(conv)::BackendLayerType>(backend)
                ->getGradientWeights();

        REQUIRE(gradientWeights.getDataDescriptor() == weightsDesc);

        Eigen::VectorXf required(1 * 1 * 3 * 3);

        required << 2400, 1200, 0, 2400, 1200, 0, 2400, 1200, 0;

        for (index_t i = 0; i < 1 * 1 * 3 * 3; ++i)
            REQUIRE(gradientWeights[i] == Approx(required[i]));
    }
}