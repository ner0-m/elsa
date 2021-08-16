#include "doctest/doctest.h"
#include <type_traits>
#include <random>
#include <iostream>

#include "elsaDefines.h"
#include "DataDescriptor.h"
#include "DnnlPoolingLayer.h"

using namespace elsa;
using namespace elsa::ml;
using namespace elsa::ml::detail;
using namespace doctest;

TEST_SUITE_BEGIN("ml-dnnl");

// TODO(dfrank): remove and replace with proper doctest usage of test cases
#define SECTION(name) DOCTEST_SUBCASE(name)

TEST_CASE("DnnlPoooling")
{
    // Example from http://cs231n.github.io/convolutional-networks/

    // Create input data
    IndexVector_t inputVec(4);
    inputVec << 1, 1, 4, 4;
    VolumeDescriptor inputDesc(inputVec);

    Eigen::VectorXf vec(1 * 1 * 4 * 4);
    // clang-format off
        vec << 1, 1, 2, 4,
               5, 6, 7, 8,
               3, 2, 1, 0,
               1, 2, 3, 4;
    // clang-format on

    DataContainer<float> input(inputDesc, vec);

    // Output descriptor
    IndexVector_t outVec(4);
    outVec << 1, 1, 2, 2;
    VolumeDescriptor outDesc(outVec);

    // Strides and pooling-window
    IndexVector_t stridesVec(2);
    stridesVec << 2, 2;

    IndexVector_t poolingVec(2);
    poolingVec << 2, 2;

    DnnlPoolingLayer<float> layer(inputDesc, outDesc, poolingVec, stridesVec);

    // Set input and compile layer
    layer.setInput(input);
    layer.compile(PropagationKind::Full);

    // Get Dnnl exection-stream
    auto engine = layer.getEngine();
    dnnl::stream s(*engine);

    layer.forwardPropagate(s);
    auto output = layer.getOutput();

    Eigen::VectorXf required(1 * 1 * 2 * 2);
    // clang-format off
    required << 6, 8,
                3, 4;
    // clang-format on

    for (int i = 0; i < 4; ++i)
        REQUIRE(output[i] == required[i]);

    layer.setOutputGradient(output);

    layer.backwardPropagate(s);
    auto inputGradient = layer.getInputGradient();

    Eigen::VectorXf requiredGradientInput(4 * 4);
    // clang-format off
    requiredGradientInput << 0, 0, 0, 0,
                             0, 6, 0, 8,
                             3, 0, 0, 0,
                             0, 0, 0, 4;
    // clang-format on

    for (int i = 0; i < inputGradient.getSize(); ++i) {
        REQUIRE(inputGradient[i] == requiredGradientInput[i]);
    }
}
TEST_SUITE_END();
