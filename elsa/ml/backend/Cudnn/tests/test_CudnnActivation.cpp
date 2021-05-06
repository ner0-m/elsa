#include "doctest/doctest.h"
#include <random>

#include "VolumeDescriptor.h"
#include "DataContainer.h"
#include "CudnnActivation.h"
#include "CudnnDataContainerInterface.h"

using namespace elsa;
using namespace elsa::ml::detail;
using namespace doctest;

// TODO(dfrank): remove and replace with proper doctest usage of test cases
#define SECTION(name) DOCTEST_SUBCASE(name)

TEST_SUITE_BEGIN("ml-cudnn");

template <typename F, typename data_t>
static void testActivation(F func, data_t alpha, const DataContainer<data_t>& input,
                           const DataContainer<data_t>& test)
{
    REQUIRE(input.getSize() == test.getSize());

    for (int i = 0; i < input.getSize(); ++i) {
        REQUIRE(func(input[i], alpha) == Approx(test[i]));
    }
}

template <typename F, typename data_t>
static void testActivationDerivative(F func, data_t alpha, const DataContainer<data_t>& input,
                                     const DataContainer<data_t>& outGrad,
                                     const DataContainer<data_t>& test)
{
    REQUIRE(input.getSize() == test.getSize());

    for (int i = 0; i < input.getSize(); ++i) {
        REQUIRE(func(input[i], alpha) * outGrad[i] == Approx(test[i]));
    }
}

template <typename LayerType, typename Func, typename FuncDer>
void testActivationLayer(Func f, FuncDer f_der)
{
    std::mt19937 mt(19937); // The random number generator using a deterministic seed
    std::uniform_real_distribution<float> dist(0, 5);
    std::uniform_int_distribution<index_t> distIdx(1, 10);

    // Create random input
    IndexVector_t inputVec(4);
    for (int i = 0; i < inputVec.size(); ++i)
        inputVec[i] = distIdx(mt);

    VolumeDescriptor inputDesc(inputVec);

    DataContainer<float> input(inputDesc);
    for (auto& coeff : input)
        coeff = dist(mt);

    // Create random output-gradient
    DataContainer<float> outputGradient(inputDesc);
    for (auto& coeff : outputGradient)
        coeff = dist(mt);

    // Construct layer and set parameters
    float alpha = dist(mt);
    LayerType layer(inputDesc, alpha);

    // Set input and compile layer
    layer.setInput(input);

    SECTION("Basics")
    {
        REQUIRE(layer.getInputDescriptor() == inputDesc);
        REQUIRE(layer.getOutputDescriptor() == inputDesc);
    }
    SECTION("Forward")
    {
        layer.forwardPropagate();
        auto output = layer.getOutput();

        testActivation(f, alpha, input, output);
    }

    SECTION("Backward")
    {
        layer.setOutputGradient(outputGradient);
        layer.backwardPropagate();
        auto inputGradient = layer.getInputGradient();

        testActivationDerivative(f_der, alpha, input, outputGradient, inputGradient);
    }
}

TEST_CASE("CudnnRelu")
{
    auto f = []([[maybe_unused]] const auto& coeff, [[maybe_unused]] float alpha) {
        if (coeff <= 0)
            return 0.f * coeff;
        else
            return coeff;
    };

    auto f_der = []([[maybe_unused]] const auto& coeff, [[maybe_unused]] float alpha) {
        if (coeff <= 0)
            return 0.f;
        else
            return 1.f;
    };
    testActivationLayer<CudnnRelu<float>>(f, f_der);
}

TEST_CASE("CudnnElu")
{
    auto f = []([[maybe_unused]] const auto& coeff, [[maybe_unused]] float alpha) {
        if (coeff <= 0)
            return alpha * (std::exp(coeff) - 1);
        else
            return coeff;
    };

    auto f_der = []([[maybe_unused]] const auto& coeff, [[maybe_unused]] float alpha) {
        if (coeff <= 0)
            return alpha * std::exp(coeff);
        else
            return 1.f;
    };
    testActivationLayer<CudnnElu<float>>(f, f_der);
}

TEST_SUITE_END();
